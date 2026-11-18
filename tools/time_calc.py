# tools/time_calc.py
import ast
import math
import operator as op
import re
from datetime import datetime
from config import TIME_KEYS

def is_time_intent(text):
    return any(k in text for k in TIME_KEYS)

def answer_time(text):
    now = datetime.now()
    if "几点" in text or "时间" in text: return f"现在是 {now:%H:%M}。"
    if "几号" in text or "日期" in text: return f"今天是 {now:%Y年%m月%d日}。"
    if "星期" in text or "周几" in text: return f"今天是星期{'一二三四五六日'[now.weekday()]}。"
    return f"现在是 {now:%Y-%m-%d %H:%M}。"

CN_NUMS = {'零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}

def cn_to_arabic(text):
    def replace_ten(m):
        s = m.group(0)
        if s == '十': return '10'
        if '十' in s:
            parts = s.split('十')
            tens = CN_NUMS.get(parts[0], 1) if parts[0] else 1
            units = CN_NUMS.get(parts[1], 0) if parts[1] else 0
            return str(tens * 10 + units)
        return s
    text = re.sub(r'[一二两三四五六七八九]?十[一二三四五六七八九]?', replace_ten, text)
    for cn, num in CN_NUMS.items(): text = text.replace(cn, str(num))
    return text

def is_calc_intent(text):
    if is_time_intent(text): return False
    has_digit = any(ch.isdigit() for ch in text) or any(cn in text for cn in CN_NUMS)
    has_op = any(op in text for op in ["+", "-", "*", "/", "×", "÷", "^", "%", "加", "减", "乘", "除", "次方", "平方根", "根号", "sin", "cos", "tan", "log"])
    return has_digit and has_op

SAFE_OPERATORS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow, ast.Mod: op.mod, ast.USub: op.neg}
SAFE_FUNCTIONS = {'sin': math.sin, 'cos': math.cos, 'tan': math.tan, 'log': math.log10, 'ln': math.log, 'sqrt': math.sqrt, 'abs': abs, 'round': round, 'pi': math.pi, 'e': math.e}


def safe_eval(node):
    # Python 3.8+ 中 ast.Num 已被移除，所有数字统一使用 ast.Constant
    if isinstance(node, ast.Constant): 
        return node.value
    elif isinstance(node, ast.BinOp): 
        return SAFE_OPERATORS[type(node.op)](safe_eval(node.left), safe_eval(node.right))
    elif isinstance(node, ast.UnaryOp): 
        return SAFE_OPERATORS[type(node.op)](safe_eval(node.operand))
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in SAFE_FUNCTIONS:
            return SAFE_FUNCTIONS[node.func.id](*[safe_eval(arg) for arg in node.args])
        raise ValueError(f"不支持函数: {node.func.id}")
    elif isinstance(node, ast.Name):
        if node.id in SAFE_FUNCTIONS: 
            return SAFE_FUNCTIONS[node.id]
        raise ValueError(f"未知变量: {node.id}")
    else: 
        raise ValueError(f"不支持语法: {type(node).__name__}")

def answer_calc(text):
    expr = cn_to_arabic(text)
    expr = expr.replace("乘以", "*").replace("乘", "*").replace("×", "*")
    expr = expr.replace("除以", "/").replace("除", "/").replace("÷", "/")
    expr = expr.replace("加上", "+").replace("加", "+").replace("减去", "-").replace("减", "-")
    
    # 1. 优先处理特殊的自然语言
    expr = expr.replace("的平方", "**2")
    expr = expr.replace("开根号", "sqrt").replace("根号", "sqrt")
    expr = re.sub(r'sqrt\s*(\d+(?:\.\d+)?)', r'sqrt(\1)', expr)
    
    # 2. 处理“A的B%” 这种表达（转化为乘法）
    expr = re.sub(r'(\d+(?:\.\d+)?)\s*的\s*(\d+(?:\.\d+)?)\s*%', r'\1*(\2/100)', expr)
    
    # 3. 处理普通的百分比（如单独的 10%）
    expr = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'(\1/100)', expr)
    
    # 4. 最后处理“的”作为乘方（如 2的10次方）
    expr = expr.replace("的", "**").replace("次方", "")

    # 清理多余词汇
    for word in ["等于多少", "等于", "计算", "算一下", "算算", "多少", "帮我", "请问"]:
        expr = expr.replace(word, "")

    # 提取合法算式
    m = re.search(r"[0-9a-zA-Z\+\-\*\/\(\)\.\s]+", expr)
    if not m: 
        return "没识别出算式，请用类似“3+5*2”的写法。"
    raw = m.group().strip()
    
    try:
        result = safe_eval(ast.parse(raw, mode='eval').body)
        if isinstance(result, float): 
            result = int(result) if result.is_integer() else round(result, 4)
        return f"{raw} = {result}"
    except Exception as e: 
        return f"这个算式我算不出来（{e}），请检查输入。"
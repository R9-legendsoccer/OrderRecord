import datetime
import gc
import os
import re
import webbrowser
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

st.set_page_config(page_title="学校订餐记录智能识别与汇总系统", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "meal_records.csv")
PRINT_HTML_FILE = os.path.join(BASE_DIR, "print_meal_sheet.html")

st.markdown(
    """
    <style>
    .img-scroll-container {
        max-height: 550px;
        overflow-y: auto;
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 6px;
        background-color: #fafafa;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==================== 1. 初始化 OCR 引擎 ====================
@st.cache_resource
def init_ocr_engine():
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_textline_orientation=True, lang="ch", use_gpu=False, show_log=False)
        return ("paddle", ocr)
    except Exception:
        try:
            import easyocr
            reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
            return ("easyocr", reader)
        except Exception as e:
            return ("none", str(e))

ocr_type, ocr_engine = init_ocr_engine()

# ==================== 2. 日期标准化、周次推算与安全存取 ====================
MEAL_ORDER_MAP = {"早餐": 1, "早": 1, "午餐": 2, "中": 2, "晚餐": 3, "晚": 3}
CN_NUMS = [
    "一", "二", "三", "四", "五", "六", "七", "八", "九", "十", 
    "十一", "十二", "十三", "十四", "十五", "十六", "十七", "十八", "十九", "二十",
    "二十一", "二十二", "二十三", "二十四", "二十五"
]

SEMESTER_OPTIONS = [
    "1年级第1学期", "1年级第2学期",
    "2年级第1学期", "2年级第2学期",
    "3年级第1学期", "3年级第2学期",
    "4年级第1学期", "4年级第2学期",
    "5年级第1学期", "5年级第2学期",
    "6年级第1学期", "6年级第2学期",
]

def format_date_str(val):
    """强制将 2026/11/9 或 2026.11.9 转为标准 2026-11-09，彻底解决日期倒序问题"""
    if pd.isna(val) or not str(val).strip():
        return ""
    s = str(val).strip().split(" ")[0].replace("/", "-").replace(".", "-").replace("年", "-").replace("月", "-").replace("日", "")
    parts = [p for p in s.split("-") if p]
    if len(parts) == 3:
        try:
            y, m, d = int(parts[0]), int(parts[1]), int(parts[2])
            return f"{y:04d}-{m:02d}-{d:02d}"
        except Exception:
            pass
    try:
        dt = pd.to_datetime(s, errors="coerce")
        if pd.notna(dt):
            return dt.strftime("%Y-%m-%d")
    except Exception:
        pass
    return s

def week_index_from_str(w_str):
    m = re.search(r'第([一二三四五六七八九十]+)周', str(w_str))
    if m and m.group(1) in CN_NUMS:
        return CN_NUMS.index(m.group(1)) + 1
    m_num = re.search(r'(\d+)', str(w_str))
    if m_num:
        return int(m_num.group(1))
    return 999

def auto_assign_week_label(date_str, start_monday=datetime.date(2026, 8, 31)):
    try:
        std_d = format_date_str(date_str)
        dt = datetime.datetime.strptime(std_d, "%Y-%m-%d").date()
        base_monday = start_monday - datetime.timedelta(days=start_monday.weekday())
        diff_days = (dt - base_monday).days
        if diff_days >= 0:
            w_num = (diff_days // 7) + 1
            if 1 <= w_num <= len(CN_NUMS):
                return f"第{CN_NUMS[w_num - 1]}周"
            return f"第{w_num}周"
    except Exception:
        pass
    return "未分类周"

def sort_meals(df):
    if df.empty:
        return df
    df_temp = df.dropna(how='all').copy()
    if "餐别" not in df_temp.columns or "日期" not in df_temp.columns:
        return df_temp
    df_temp["日期"] = df_temp["日期"].apply(format_date_str)
    df_temp["_dt"] = pd.to_datetime(df_temp["日期"], errors="coerce")
    df_temp["_meal_weight"] = df_temp["餐别"].map(
        lambda x: MEAL_ORDER_MAP.get(str(x).strip(), 4) if pd.notna(x) else 4
    )
    df_temp = df_temp.sort_values(by=["_dt", "_meal_weight"]).drop(columns=["_dt", "_meal_weight"]).reset_index(drop=True)
    return df_temp

def normalize_dataframe(df, default_semester="1年级第1学期", start_monday=datetime.date(2026, 8, 31)):
    df = df.dropna(how='all').copy()
    
    if "日期" in df.columns:
        df["日期"] = df["日期"].apply(format_date_str)
    else:
        df["日期"] = ""
        
    # 强制将学期和周次转为干净的字符串，防止出现 float 类型冲突
    if "学期" not in df.columns:
        df["学期"] = default_semester
    else:
        df["学期"] = df["学期"].apply(
            lambda x: str(x).strip() if (pd.notna(x) and str(x).strip() and str(x).strip() != "nan") else default_semester
        )

    if "周次" not in df.columns:
        if "备注" in df.columns:
            df["周次"] = df["备注"].apply(lambda x: str(x).strip() if "周" in str(x) else "")
        else:
            df["周次"] = ""
            
    df["周次"] = [
        str(w).strip() if (pd.notna(w) and str(w).strip() and "周" in str(w) and str(w).strip() != "nan" and str(w).strip() != "未分类周") 
        else auto_assign_week_label(d, start_monday)
        for d, w in zip(df["日期"], df["周次"])
    ]
    
    if "状态" not in df.columns:
        df["状态"] = "取餐"
    else:
        def clean_status(val):
            s = str(val).strip()
            if "退" in s:
                return "退餐"
            elif "未" in s:
                return "未取餐"
            return "取餐"
        df["状态"] = df["状态"].apply(clean_status)

    if "餐食评价" not in df.columns:
        df["餐食评价"] = df["是否好吃"] if "是否好吃" in df.columns else "一般"
            
    def clean_rating(val):
        s = str(val).strip()
        if s in ["好吃", "好"]:
            return "好"
        elif s in ["踩雷", "不好", "差"]:
            return "不好"
        return "一般"
    df["餐食评价"] = df["餐食评价"].apply(clean_rating)
    
    df["价格(元)"] = pd.to_numeric(df.get("价格(元)", 0.0), errors="coerce").fillna(0.0)
    
    target_cols = ["学期", "周次", "日期", "餐别", "菜品种类", "价格(元)", "状态", "餐食评价"]
    for col in target_cols:
        if col not in df.columns:
            df[col] = ""
        # 确保所有列都是合法的 string/float 类型
        if col in ["学期", "周次", "日期", "餐别", "菜品种类", "状态", "餐食评价"]:
            df[col] = df[col].astype(str).replace("nan", "")
            
    return sort_meals(df[target_cols])

def load_data():
    if os.path.exists(DATA_FILE):
        for enc in ['utf-8-sig', 'utf-8', 'gbk', 'gb18030']:
            try:
                raw_df = pd.read_csv(DATA_FILE, encoding=enc)
                norm_df = normalize_dataframe(raw_df)
                # 自动将标准化后的日期（如 2026/11/9 -> 2026-11-09）写回文件
                norm_df.to_csv(DATA_FILE, index=False, encoding='utf-8-sig')
                return norm_df
            except Exception:
                continue
    return pd.DataFrame(columns=["学期", "周次", "日期", "餐别", "菜品种类", "价格(元)", "状态", "餐食评价"])

def save_data(df, default_semester="1年级第1学期", start_monday=datetime.date(2026, 8, 31)):
    """全量保存函数：确保写入的是完整的历史总表"""
    norm_df = normalize_dataframe(df, default_semester, start_monday)
    norm_df.to_csv(DATA_FILE, index=False, encoding='utf-8-sig')
    return norm_df

df_meals = load_data()

if "pending_records" not in st.session_state:
    st.session_state.pending_records = None
if "duplicate_dates" not in st.session_state:
    st.session_state.duplicate_dates = []

# ==================== 3. 独立纯净打印页面生成器（100% 弹窗打印） ====================
def generate_and_open_print_page(print_df, scope_title):
    valid_df = print_df[print_df["状态"] != "退餐"]
    refund_df = print_df[print_df["状态"] == "退餐"]
    actual_total = valid_df["价格(元)"].sum()
    refund_total = refund_df["价格(元)"].sum()

    rows_html = ""
    for _, r in print_df.iterrows():
        is_refund = (r["状态"] == "退餐")
        tr_class = " class='refund-row'" if is_refund else ""
        price_str = f"¥ {float(r['价格(元)']):.2f}" + (" (退餐不计)" if is_refund else "")
        rows_html += (
            f"<tr{tr_class}>"
            f"<td class='center'>{r['学期']}</td>"
            f"<td class='center'>{r['周次']}</td>"
            f"<td class='center'>{r['日期']}</td>"
            f"<td class='center'>{r['餐别']}</td>"
            f"<td>{r['菜品种类']}</td>"
            f"<td class='right'>{price_str}</td>"
            f"<td class='center'>{r['状态']}</td>"
            f"<td class='center'>{r['餐食评价']}</td>"
            f"</tr>\n"
        )

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>学校订餐消费记录明细表 - {scope_title}</title>
    <style>
        body {{
            font-family: "Microsoft YaHei", "SimSun", sans-serif;
            margin: 20px;
            color: #000;
        }}
        .no-print-bar {{
            background: #eef4ff;
            border: 1px solid #b6d0ff;
            padding: 10px 16px;
            margin-bottom: 16px;
            border-radius: 6px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .btn-print {{
            background: #2b5cff;
            color: #fff;
            border: none;
            padding: 8px 18px;
            border-radius: 5px;
            font-size: 14px;
            font-weight: bold;
            cursor: pointer;
        }}
        h2 {{
            text-align: center;
            margin: 0 0 8px 0;
            font-size: 20px;
        }}
        .summary-line {{
            text-align: center;
            font-size: 13.5px;
            margin-bottom: 14px;
            color: #222;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            border: 1px solid #222;
            padding: 6px 8px;
            font-size: 12.5px;
        }}
        th {{
            background-color: #f2f2f2;
            text-align: center;
        }}
        .center {{ text-align: center; }}
        .right {{ text-align: right; }}
        .refund-row {{
            color: #777;
            text-decoration: line-through;
        }}
        @media print {{
            .no-print-bar {{ display: none !important; }}
            body {{ margin: 5mm; }}
        }}
    </style>
</head>
<body onload="setTimeout(function(){{ window.print(); }}, 300);">
    <div class="no-print-bar">
        <span>📄 <strong>打印预览页已就绪</strong>：如果浏览器未自动弹出打印窗口，请点击右侧按钮或按 <strong>Ctrl + P</strong></span>
        <button class="btn-print" onclick="window.print()">🖨️ 立即打印 / 另存为 PDF</button>
    </div>
    <h2>学校订餐消费记录明细表（{scope_title}）</h2>
    <div class="summary-line">
        <strong>有效用餐：</strong>{len(valid_df)} 餐 &nbsp;|&nbsp;
        <strong>实际消费总额（不含退餐）：¥ {actual_total:.2f}</strong> &nbsp;|&nbsp;
        <strong>退餐：</strong>{len(refund_df)} 餐（¥ {refund_total:.2f}） &nbsp;|&nbsp;
        <strong>生成日期：</strong>{datetime.date.today()}
    </div>
    <table>
        <thead>
            <tr>
                <th style="width: 11%;">学期</th>
                <th style="width: 8%;">周次</th>
                <th style="width: 11%;">日期</th>
                <th style="width: 7%;">餐别</th>
                <th style="width: 39%;">菜品种类 / 食堂</th>
                <th style="width: 10%;">价格</th>
                <th style="width: 7%;">状态</th>
                <th style="width: 7%;">评价</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
</body>
</html>"""

    with open(PRINT_HTML_FILE, "w", encoding="utf-8") as f:
        f.write(html_content)
    webbrowser.open(f"file:///{PRINT_HTML_FILE.replace(os.sep, '/')}")
    return html_content

# ==================== 4. OpenCV 切片与 OCR 解析 ====================
STOP_WORDS = [
    "未取餐", "末取餐", "已取餐", "己取餐", "已退餐", "己退餐", "退餐", "退」", 
    "自提", "点餐", "总价", "小计", "小计金额", "订单详情", "订单状态", "订单明细", 
    "基本信息", "支付方式", "配送方式", "余额支付", "堂食", "手机号", "支付时间", 
    "去结算", "提交订单", "浪餐", "浪答", "订泮", "末取答", "已", "己", "卣"
]

def detect_meal_status_by_image(img_rgb, y_top, y_bottom):
    h, w, _ = img_rgb.shape
    y0 = max(0, int(y_top))
    y1 = min(h, int(y_bottom))
    if y1 <= y0 + 5:
        return None
    roi = img_rgb[y0:y1, int(w * 0.14):int(w * 0.35), :]
    green_mask = (roi[:, :, 1].astype(int) - roi[:, :, 0].astype(int) > 30) & (roi[:, :, 1] > 130)
    if np.sum(green_mask) > 25:
        return "green"
    r, g, b = roi[:, :, 0].astype(int), roi[:, :, 1].astype(int), roi[:, :, 2].astype(int)
    grey_mask = (np.abs(r - g) < 12) & (np.abs(g - b) < 12) & (r > 120) & (r < 195)
    if np.sum(grey_mask) > 35:
        return "grey"
    return None

def run_sliced_ocr(pil_image):
    img_rgb = np.array(pil_image.convert("RGB"))
    orig_h, orig_w, _ = img_rgb.shape
    
    target_w = 900
    scale = target_w / float(orig_w)
    target_h = int(orig_h * scale)
    img_resized = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    
    slice_h = 1400
    overlap = 150
    all_items = []
    
    y_start = 0
    while y_start < target_h:
        y_end = min(target_h, y_start + slice_h)
        chunk = img_resized[y_start:y_end, :, :]
        
        if ocr_type == "paddle":
            res = ocr_engine.ocr(chunk, cls=True)
            if res and res[0]:
                for line in res[0]:
                    box, (txt, score) = line[0], line[1]
                    if score < 0.45:
                        continue
                    all_items.append((box[0][1] + y_start, box[0][0], txt.strip()))
        elif ocr_type == "easyocr":
            res = ocr_engine.readtext(chunk)
            for box, txt, score in res:
                if score < 0.35:
                    continue
                all_items.append((box[0][1] + y_start, box[0][0], txt.strip()))
                
        if y_end >= target_h:
            break
        y_start += (slice_h - overlap)
        
    all_items.sort(key=lambda x: (x[0], x[1]))
    deduped = []
    for item in all_items:
        if deduped and abs(item[0] - deduped[-1][0]) < 12 and item[2] == deduped[-1][2]:
            continue
        deduped.append(item)
        
    return deduped, img_resized

def format_package_colon(text):
    # 1. 修复字母套餐漏字与形近字（如 D、G、B、L、E 等字母被吞或识别成“匕”）
    text = re.sub(r'匕\s*套餐', 'L套餐', text)
    text = re.sub(r'[」』]\s*套餐', 'J套餐', text)
    
    # 如果文本包含“套餐”但前面没有英文字母，根据特征或常见漏字进行智能补齐
    if "套餐" in text and not re.search(r'[A-Za-z]\s*套餐', text):
        if "菠萝咕" in text or "咕咾肉" in text or "滑溜里脊" in text:
            text = text.replace("套餐", "E套餐", 1)
        elif "糖三角" in text or "煎饺" in text:
            text = text.replace("套餐", "C套餐", 1)
        elif "馅饼" in text or "酱香饼" in text or "灌饼" in text:
            text = text.replace("套餐", "B套餐", 1)
        elif "锅包肉" in text or "大盘鸡" in text:
            text = text.replace("套餐", "L套餐", 1)
        elif "咖喱" in text or "羊排" in text or "牛腩" in text:
            text = text.replace("套餐", "M套餐", 1)

    # 2. 深度修复数量词漏掉数字的硬伤（如 “(张)”变成“(2张)”，“个”前面丢数字）
    text = text.replace("(张)", "(2张)").replace("( 张)", "(2张)")
    text = text.replace("(份)", "(2份)").replace("( 份)", "(2份)")
    
    # 修复“水煮蛋1个”被吞成“水煮蛋个”
    text = re.sub(r'水煮蛋\s*个', '水煮蛋1个', text)
    text = re.sub(r'手抓饼\s*份', '手抓饼1份', text)
    text = re.sub(r'手抓饼\s*个', '手抓饼1个', text)

    # 修复带数字的括号（如把 "(2个)" 误认成 "(2个)" 或 “(8个)”/“(20个)” 被吞数字的）
    # 针对特定大份水饺（猪肉白菜馅20个、三鲜馅20个）进行兜底
    if "猪肉白菜馅" in text and "20" not in text:
        text = text.replace("猪肉白菜馅个", "猪肉白菜馅20个").replace("猪肉白菜馅", "猪肉白菜馅20个")
    if "猪肉三鲜馅" in text and "20" not in text:
        text = text.replace("猪肉三鲜馅个", "猪肉三鲜馅20个").replace("猪肉三鲜馅", "猪肉三鲜馅20个")

    # 3. 清理冗余符号
    text = re.sub(r'\*1\b', '', text)
    text = re.sub(r'([A-Za-z0-9]套餐(\([^)]+\))?)\s*[:：=＝]?\s*', r'\1: ', text)
    return text.strip()

def clean_dish_text(text):
    for w in STOP_WORDS:
        text = text.replace(w, " ")
    for loc in ["第二食堂", "笫二食堂", "二食堂", "第一食堂", "一食堂"]:
        text = text.replace(loc, " ")
    
    # 4. 修复“干”被 OCR 误认成“千”的固定硬伤
    text = text.replace("千煸", "干煸").replace("千煸", "干煸").replace("千锅", "干锅")
    
    # 5. 统一标点与空格
    text = text.replace("、", "+").replace(",", "+").replace("，", "+")
    text = re.sub(r'[¥￥]?\s*\d+(\.\d{1,2})?', '', text)
    text = re.sub(r'[:：;,，=＝\s\t\n半\\|*]+', ' ', text).strip()
    text = re.sub(r'\s*\+\s*', '+', text)
    
    return format_package_colon(text)

def parse_price(raw_text):
    match_decimal = re.findall(r'(\d+\.\d{1,2})', raw_text)
    if match_decimal:
        try:
            val = float(match_decimal[-1])
            return 7.5 if val == 7.3 else val
        except Exception:
            pass
    match_int = re.findall(r'(\d+)', raw_text)
    if match_int:
        try:
            val = int(match_int[-1])
            if 5 <= val <= 35:
                return float(val)
            elif 500 <= val <= 3500 and val % 100 == 0:
                return float(val / 100)
        except Exception:
            pass
    return 0.0

def extract_base_monday(ocr_items):
    date_regex = re.compile(r'202\d[-/.年]\d{1,2}[-/.月]\d{1,2}')
    for _, _, text in ocr_items[:20]:
        if any(kw in text for kw in ["下单时间", "支付时间"]):
            m = date_regex.search(text)
            if m:
                try:
                    d_str = format_date_str(m.group(0))
                    order_date = datetime.datetime.strptime(d_str, "%Y-%m-%d").date()
                    if order_date.weekday() == 5:
                        return order_date + datetime.timedelta(days=9)
                    elif order_date.weekday() == 6:
                        return order_date + datetime.timedelta(days=8)
                    else:
                        days_ahead = (7 - order_date.weekday()) % 7
                        return order_date + datetime.timedelta(days=days_ahead if days_ahead > 0 else 7)
                except Exception:
                    pass
    return None

def parse_sliced_ocr(ocr_items, img_resized, current_semester, semester_start_monday):
    records = []
    date_regex = re.compile(r'202\d[-/.年]\d{1,2}[-/.月]\d{1,2}')
    
    current_date = ""
    current_meal = None
    meal_start_y = 0
    item_buffer = []

    def commit_meal(end_y):
        if not current_meal or not item_buffer or not current_date:
            return
        full_text = " ".join(item_buffer)
        price = parse_price(full_text)
        
        color_tag = detect_meal_status_by_image(img_resized, meal_start_y, end_y)
        if "已退餐" in full_text or "己退餐" in full_text or color_tag == "grey":
            status = "退餐"
        elif "未取餐" in full_text or "末取餐" in full_text:
            status = "未取餐"
        else:
            status = "取餐"
            
        is_second = any(loc in full_text for loc in ["第二食堂", "笫二食堂", "二食堂"])
        canteen_tag = "[第二食堂] " if is_second else "[第一食堂] "
        clean_text = clean_dish_text(full_text)
        
        if len(clean_text) < 3 or re.search(r'202\d|订单|支付|^\d+$', clean_text):
            return
            
        records.append({
            "学期": current_semester,
            "周次": "",
            "日期": format_date_str(current_date), # 严格使用截图中读出来的原始日期
            "餐别": current_meal,
            "菜品种类": f"{canteen_tag}{clean_text}",
            "价格(元)": price,
            "状态": status,
            "餐食评价": "一般"
        })

    for y, x, text in ocr_items:
        if any(h in text for h in ["订单详情", "下单时间", "支付时间", "订单编号", "订单金额"]):
            continue
            
        # 1. 严格捕获截图中自带的日期
        d_match = date_regex.search(text)
        if d_match:
            commit_meal(y)
            current_meal = None
            item_buffer = []
            current_date = d_match.group(0) # 锁定当前识别到的日期，绝不自动顺延！
            continue
            
        # 2. 捕获餐别
        found_meal = None
        for m in ["早餐", "午餐", "晚餐"]:
            if m in text:
                found_meal = m
                break
        if found_meal:
            commit_meal(y)
            current_meal = found_meal
            meal_start_y = y
            item_buffer = []
            rem = text.replace(found_meal, "").strip()
            if rem:
                item_buffer.append(rem)
            continue
            
        if current_meal:
            item_buffer.append(text)

    commit_meal(img_resized.shape[0])
    
    # 3. 仅根据最终确定的日期，自动匹配对应的周次标签
    for r in records:
        r["周次"] = auto_assign_week_label(r["日期"], semester_start_monday)
        
    return records

# ==================== 5. 主界面布局 ====================
st.title("🍽️ 学校订餐记录智能识别与汇总系统")
engine_badge = "PaddleOCR 高精切片引擎" if ocr_type == "paddle" else "EasyOCR 兼容引擎"
st.caption(f"当前识别内核：**{engine_badge}** | 唯一行号保护：筛选修改单条数据绝不丢失其余记录 | 汇总自动剔除「退餐」")

col_img, col_table = st.columns([1, 1.9], gap="medium")

with col_img:
    st.subheader("⚙️ 学期设置与长图上传")
    
    with st.expander("🎓 当前学期与第一周起始日设置", expanded=True):
        current_semester = st.selectbox("当前录入的学期：", options=SEMESTER_OPTIONS, index=0)
        semester_start_monday = st.date_input(
            "本学期【第一周】周一日期（用于算周次）：",
            value=datetime.date(2026, 8, 31)
        )
    
    uploaded_file = st.file_uploader("上传整周订餐长截图", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.markdown('<div class="img-scroll-container">', unsafe_allow_html=True)
        st.image(image, caption="长图预览（框内上下滚动对照）", width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🚀 开始高精度识别（自动归入学期与周次）", type="primary"):
            with st.spinner("正在分块扫描图片并自动计算第几周..."):
                try:
                    ocr_items, img_resized = run_sliced_ocr(image)
                    new_records = parse_sliced_ocr(
                        ocr_items, img_resized, current_semester, semester_start_monday
                    )
                    gc.collect()
                    
                    if new_records:
                        detected_week = new_records[0]["周次"]
                        existing_dates = set(df_meals["日期"].unique()) if not df_meals.empty else set()
                        new_dates = set(r["日期"] for r in new_records)
                        overlap = sorted(list(existing_dates & new_dates))
                        
                        if overlap:
                            st.session_state.pending_records = new_records
                            st.session_state.duplicate_dates = overlap
                        else:
                            new_df = pd.DataFrame(new_records)
                            df_meals = pd.concat([df_meals, new_df], ignore_index=True)
                            df_meals = save_data(df_meals, current_semester, semester_start_monday)
                            st.success(f"成功导入【{current_semester} · {detected_week}】共 {len(new_records)} 餐记录！")
                            st.rerun()
                    else:
                        st.warning("未能识别出有效餐品，请检查截图是否完整。")
                except Exception as e:
                    st.error(f"识别出错: {e}")
    else:
        st.info("上传长截图后，系统将自动切分菜品并归入【第X周】。")

# ==================== 6. 右侧：核对、独立页打印与防丢保存 ====================
with col_table:
    st.subheader("📋 订餐明细核对与按周打印")

    # 重复日期拦截提示
    if st.session_state.pending_records:
        dup_str = "、".join(st.session_state.duplicate_dates)
        detected_w = st.session_state.pending_records[0].get("周次", "")
        st.warning(f"⚠️ **检测到【{detected_w}】以下日期已存在记录：** {dup_str}")
        c_dup1, c_dup2, c_dup3 = st.columns(3)
        with c_dup1:
            if st.button("🔄 覆盖已有日期", type="primary"):
                df_meals = df_meals[~df_meals["日期"].isin(st.session_state.duplicate_dates)]
                df_meals = pd.concat([df_meals, pd.DataFrame(st.session_state.pending_records)], ignore_index=True)
                df_meals = save_data(df_meals, current_semester, semester_start_monday)
                st.session_state.pending_records = None
                st.rerun()
        with c_dup2:
            if st.button("➕ 仅追加新日期"):
                filtered = [r for r in st.session_state.pending_records if r["日期"] not in st.session_state.duplicate_dates]
                if filtered:
                    df_meals = pd.concat([df_meals, pd.DataFrame(filtered)], ignore_index=True)
                    df_meals = save_data(df_meals, current_semester, semester_start_monday)
                st.session_state.pending_records = None
                st.rerun()
        with c_dup3:
            if st.button("❌ 取消本次导入"):
                st.session_state.pending_records = None
                st.rerun()

    # 为全局总表分配唯一的行追踪号 _row_id（确保部分筛选保存时，100% 精准对号入座，绝不误删其他周数据）
    df_meals_with_id = df_meals.copy()
    df_meals_with_id["_row_id"] = df_meals_with_id.index

    # 多级筛选
    s_col1, s_col2, s_col3 = st.columns([1.1, 1.1, 1.5])
    all_semesters = sorted(df_meals_with_id["学期"].unique()) if not df_meals_with_id.empty else ["1年级第1学期"]
    with s_col1:
        sel_semester = st.selectbox("1️⃣ 选择学期：", options=["全部学期"] + all_semesters, index=1 if len(all_semesters) == 1 else 0)
        
    sem_filtered_df = df_meals_with_id if sel_semester == "全部学期" else df_meals_with_id[df_meals_with_id["学期"] == sel_semester]
    all_weeks_sorted = sorted(sem_filtered_df["周次"].unique(), key=week_index_from_str) if not sem_filtered_df.empty else []

    with s_col2:
        filter_mode = st.radio("2️⃣ 筛选方式：", ["按周次选择", "按日期范围"], horizontal=False)
        
    with s_col3:
        sem_label = "" if sel_semester == "全部学期" else f"{sel_semester} · "
        if filter_mode == "按周次选择" and all_weeks_sorted:
            week_options = ["全部周次"] + all_weeks_sorted
            selected_week = st.selectbox("3️⃣ 选择周次（支持单周打印）：", options=week_options, index=0)
            if selected_week == "全部周次":
                display_df = sem_filtered_df.copy()
                print_scope_title = f"{sem_label}全部周次（共 {len(all_weeks_sorted)} 周）"
            else:
                display_df = sem_filtered_df[sem_filtered_df["周次"] == selected_week].copy()
                d_sorted = pd.to_datetime(display_df['日期'], errors='coerce').dropna().sort_values()
                start_fmt = d_sorted.iloc[0].strftime("%Y-%m-%d") if not d_sorted.empty else ""
                end_fmt = d_sorted.iloc[-1].strftime("%Y-%m-%d") if not d_sorted.empty else ""
                print_scope_title = f"{sem_label}{selected_week}（{start_fmt} ~ {end_fmt}）"
        else:
            date_series = pd.to_datetime(sem_filtered_df["日期"], errors="coerce").dropna()
            min_d = date_series.min().date() if not date_series.empty else datetime.date.today()
            max_d = date_series.max().date() if not date_series.empty else datetime.date.today()
            sel_range = st.date_input("3️⃣ 选择起止日期：", value=(min_d, max_d))
            if isinstance(sel_range, (tuple, list)) and len(sel_range) == 2:
                s_d, e_d = sel_range
                mask = (pd.to_datetime(sem_filtered_df["日期"], errors="coerce").dt.date >= s_d) & \
                       (pd.to_datetime(sem_filtered_df["日期"], errors="coerce").dt.date <= e_d)
                display_df = sem_filtered_df[mask].copy()
                print_scope_title = f"{sem_label}{s_d} ~ {e_d}"
            else:
                display_df = sem_filtered_df.copy()
                print_scope_title = f"{sem_label}全部记录"

    # 统计卡片（严格剔除退餐）
    if not display_df.empty:
        valid_df = display_df[display_df["状态"] != "退餐"]
        refund_df = display_df[display_df["状态"] == "退餐"]
        actual_spend = valid_df["价格(元)"].sum()
        refund_spend = refund_df["价格(元)"].sum()
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("当前有效消费（不含退餐）", f"¥ {actual_spend:.2f}")
        m2.metric("有效餐数 / 当前显示", f"{len(valid_df)} 餐 / {len(display_df)} 餐")
        m3.metric("退餐笔数与金额", f"{len(refund_df)} 餐 (¥ {refund_spend:.2f})")
        good_cnt = len(valid_df[valid_df["餐食评价"] == "好"])
        bad_cnt = len(valid_df[valid_df["餐食评价"] == "不好"])
        m4.metric("评价（好 / 不好）", f"👍 {good_cnt}  |  👎 {bad_cnt}")

    # 交互式数据编辑器（隐藏 _row_id，用它追踪每一行在总表中的位置）
    edited_df = st.data_editor(
        display_df,
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
        key="meal_editor",
        column_order=["学期", "周次", "日期", "餐别", "菜品种类", "价格(元)", "状态", "餐食评价"],
        column_config={
            "_row_id": None,
            "学期": st.column_config.SelectboxColumn("学期", options=SEMESTER_OPTIONS, required=True),
            "周次": st.column_config.TextColumn("周次", required=True),
            "日期": st.column_config.TextColumn("日期", required=True),
            "餐别": st.column_config.SelectboxColumn("餐别", options=["早餐", "午餐", "晚餐"], required=True),
            "菜品种类": st.column_config.TextColumn("菜品种类", width="large"),
            "价格(元)": st.column_config.NumberColumn("价格(元)", format="¥ %.2f", min_value=0.0, step=0.5),
            "状态": st.column_config.SelectboxColumn("状态", options=["未取餐", "取餐", "退餐"], required=True),
            "餐食评价": st.column_config.SelectboxColumn("餐食评价", options=["好", "一般", "不好"], required=True),
        }
    )

    # 操作按钮栏：保存修改、一键打开独立打印窗口、校准周次
    b_col1, b_col2, b_col3 = st.columns([1.2, 1.2, 1.1])
    with b_col1:
        if st.button("💾 保存当前表格修改（全表安全合并）", type="primary"):
            # 1. 保留总表中所有【不在当前筛选视图内】的其他周数据
            shown_ids = set(display_df["_row_id"].dropna().astype(int))
            unshown_df = df_meals_with_id[~df_meals_with_id["_row_id"].isin(shown_ids)].drop(columns=["_row_id"], errors="ignore")
            
            # 2. 获取当前筛选视图编辑后的数据（包含修改后的行、新增的行，且自动剔除被用户删除的行）
            current_edited_clean = edited_df.drop(columns=["_row_id"], errors="ignore")
            
            # 3. 将未显示的其余所有历史数据 + 当前修改后的数据 完整合并，写回 CSV
            merged_full_df = pd.concat([unshown_df, current_edited_clean], ignore_index=True)
            df_meals = save_data(merged_full_df, current_semester, semester_start_monday)
            st.success(f"✅ 保存成功！当前共保存 {len(df_meals)} 条完整记录到 meal_records.csv，其余周次数据完好无损。")
            st.rerun()

    with b_col2:
        if st.button("🖨️ 打开独立打印页面（支持按周打印）"):
            if not display_df.empty:
                generate_and_open_print_page(display_df, print_scope_title)
                st.success("✅ 已在浏览器新标签页打开专属打印页并自动调起打印窗口！")
            else:
                st.warning("当前筛选范围内没有可打印的数据。")

    with b_col3:
        if st.button("🔄 重新校准所有日期与周次"):
            df_meals["日期"] = df_meals["日期"].apply(format_date_str)
            df_meals["周次"] = [auto_assign_week_label(d, semester_start_monday) for d in df_meals["日期"]]
            df_meals = save_data(df_meals, current_semester, semester_start_monday)
            st.success("已将所有日期统一为 YYYY-MM-DD 并校准周次！")
            st.rerun()

# ==================== 7. 底部：每周统计汇总表 ====================
st.markdown("---")
st.subheader("📊 每周消费统计汇总表（汇总金额已自动剔除退餐）")

if not df_meals.empty:
    weekly_stats = []
    for sem_name in sorted(df_meals["学期"].unique()):
        sem_df = df_meals[df_meals["学期"] == sem_name]
        for w_name in sorted(sem_df["周次"].unique(), key=week_index_from_str):
            w_df = sem_df[sem_df["周次"] == w_name]
            w_valid = w_df[w_df["状态"] != "退餐"]
            w_refund = w_df[w_df["状态"] == "退餐"]
            
            actual_sum = w_valid["价格(元)"].sum()
            valid_cnt = len(w_valid)
            
            # 先清洗日期再转为 datetime 排序，彻底杜绝 2026/11/10 ~ 2026/11/9 倒序
            clean_dates = w_df["日期"].apply(format_date_str)
            dt_sorted = pd.to_datetime(clean_dates, format="%Y-%m-%d", errors="coerce").dropna().sort_values()
            if not dt_sorted.empty:
                date_range_str = f"{dt_sorted.iloc[0].strftime('%Y-%m-%d')} ~ {dt_sorted.iloc[-1].strftime('%Y-%m-%d')}"
            else:
                date_range_str = "-"
            
            weekly_stats.append({
                "学期": sem_name,
                "周次": w_name,
                "日期范围": date_range_str,
                "总订餐数": len(w_df),
                "有效餐数": valid_cnt,
                "退餐数": len(w_refund),
                "实际消费金额(元)": round(actual_sum, 2),
                "退餐金额(元)": round(w_refund["价格(元)"].sum(), 2),
                "有效餐均价(元)": round(actual_sum / valid_cnt, 2) if valid_cnt > 0 else 0.0,
                "评价: 好": len(w_valid[w_valid["餐食评价"] == "好"]),
                "评价: 一般": len(w_valid[w_valid["餐食评价"] == "一般"]),
                "评价: 不好": len(w_valid[w_valid["餐食评价"] == "不好"]),
            })
        
    df_weekly_summary = pd.DataFrame(weekly_stats)
    
    all_valid = sem_filtered_df[sem_filtered_df["状态"] != "退餐"]
    all_refund = sem_filtered_df[sem_filtered_df["状态"] == "退餐"]
    
    g1, g2, g3, g4 = st.columns(4)
    g1.metric(f"🌟 {sel_semester} · 实际总支出（不含退餐）", f"¥ {all_valid['价格(元)'].sum():.2f}")
    g2.metric("🍱 累计有效总餐数", f"{len(all_valid)} 餐（共 {len(df_weekly_summary)} 周）")
    g3.metric("↩️ 累计退餐总计", f"{len(all_refund)} 餐（¥ {all_refund['价格(元)'].sum():.2f}）")
    g4.metric("💰 历史有效餐均价", f"¥ {all_valid['价格(元)'].mean():.2f}" if not all_valid.empty else "¥ 0.00")

    st.dataframe(
        df_weekly_summary,
        width="stretch",
        hide_index=True,
        column_config={
            "实际消费金额(元)": st.column_config.NumberColumn("实际消费金额 (不含退餐)", format="¥ %.2f"),
            "退餐金额(元)": st.column_config.NumberColumn("退餐金额", format="¥ %.2f"),
            "有效餐均价(元)": st.column_config.NumberColumn("有效餐均价", format="¥ %.2f"),
        }
    )
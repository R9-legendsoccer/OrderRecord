import streamlit as st
import pandas as pd
import os
from PIL import Image
import easyocr
import datetime

# 页面配置
st.set_page_config(page_title="学校订餐记录智能识别与汇总", layout="wide")

DATA_FILE = "meal_records.csv"

# 初始化 OCR 引擎（加缓存避免重复加载）
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['ch_sim', 'en'])

with st.spinner("正在加载 AI 识图引擎，请稍候..."):
    ocr_reader = load_ocr()

# 餐别排序权重映射
MEAL_ORDER_MAP = {"早餐": 1, "早": 1, "午餐": 2, "中": 2, "晚餐": 3, "晚": 3}

def sort_meals(df):
    if df.empty:
        return df
    # 增加一个临时的排序列
    df_temp = df.copy()
    df_temp["_meal_weight"] = df_temp["餐别"].astype(str).map(lambda x: MEAL_ORDER_MAP.get(x.strip(), 4))
    
    # 按日期和餐别权重升序排序
    df_temp = df_temp.sort_values(by=["日期", "_meal_weight"]).drop(columns=["_meal_weight"]).reset_index(drop=True)
    return df_temp

# 加载数据
def load_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        return sort_meals(df)
    else:
        return pd.DataFrame(columns=[
            "日期", "餐别", "用餐地点", "菜品种类", "价格(元)", "是否好吃", "备注"
        ])

def save_data(df):
    # 保存前自动排序
    sorted_df = sort_meals(df)
    sorted_df.to_csv(DATA_FILE, index=False)
    return sorted_df

df_meals = load_data()

st.title("🍽️ 学校订餐记录智能识别与核对系统")
st.markdown("上传每周订餐截图，AI 自动识别文字并智能分类；左侧看图对照，右侧表格随时核对、修改并标记美味！")

# 左右分栏布局：左边看图，右边表格
col_img, col_table = st.columns([1, 1.5], gap="medium")

with col_img:
    st.subheader("🖼️ 原图对照区")
    uploaded_file = st.file_uploader("上传本周订餐记录截图", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="当前核对的订餐图片", use_column_width=True)
        
        if st.button("🚀 开始 AI 提取识别", type="primary"):
            with st.spinner("AI 正在扫描图片中的文字..."):
                temp_path = "temp_meal.png"
                image.save(temp_path)
                
                # 执行 OCR
                results = ocr_reader.readtext(temp_path)
                extracted_texts = [text[1] for text in results]
                full_raw_text = " ".join(extracted_texts)
                
                # 智能识别用餐地点：包含“第二食堂”就是第二食堂，否则默认第一食堂
                location = "第二食堂" if "第二食堂" in full_raw_text else "第一食堂"
                
                # 智能推测餐别
                meal_type = "午餐"
                if "早餐" in full_raw_text or "早" in full_raw_text:
                    meal_type = "早餐"
                elif "晚餐" in full_raw_text or "晚" in full_raw_text:
                    meal_type = "晚餐"

                # 生成新记录
                new_row = {
                    "日期": str(datetime.date.today()),
                    "餐别": meal_type,
                    "用餐地点": location,
                    "菜品种类": " / ".join(extracted_texts[:10]), 
                    "价格(元)": 0.0,
                    "是否好吃": "一般",
                    "备注": "AI自动识别，请对照左侧原图核对"
                }
                
                df_meals = pd.concat([df_meals, pd.DataFrame([new_row])], ignore_index=True)
                df_meals = save_data(df_meals)
                st.success("识别成功！已添加至右侧表格，并已按时间排序。")
                st.rerun()
                
        if 'full_raw_text' in locals() and full_raw_text:
            with st.expander("🔍 查看 AI 识别到的全部原始文字"):
                st.text(full_raw_text)
    else:
        st.info("请先上传左侧的订餐截图。")

with col_table:
    st.subheader("📊 识别结果与核对修改表")
    st.markdown("💡 **提示**：系统会自动按日期和早/中/晚餐排序。直接双击修改文字、价格和评价即可。")
    
    # 交互式表格
    edited_df = st.data_editor(
        df_meals,
        num_rows="dynamic",
        use_container_width=True,
        key="meal_editor",
        column_config={
            "餐别": st.column_config.SelectboxColumn(
                "餐别",
                options=["早餐", "午餐", "晚餐"],
                required=True
            ),
            "用餐地点": st.column_config.SelectboxColumn(
                "用餐地点",
                options=["第一食堂", "第二食堂"],
                required=True
            ),
            "是否好吃": st.column_config.SelectboxColumn(
                "评价",
                help="标记这顿饭好不好吃",
                options=["好吃", "一般", "踩雷"],
                required=True
            ),
            "价格(元)": st.column_config.NumberColumn(
                "价格 (元)",
                format="¥ %.2f",
                min_value=0.0,
                step=0.5
            )
        }
    )

    if st.button("💾 保存所有修改与备注", type="primary"):
        df_meals = save_data(edited_df)
        st.success("修改已成功同步、排序并永久保存！")
        st.rerun()

# 底部汇总统计与红黑榜
st.markdown("---")
st.header("📈 消费统计与美食红黑榜")

if not df_meals.empty:
    c1, c2, c3 = st.columns(3)
    total_spent = df_meals["价格(元)"].astype(float).sum()
    total_meals = len(df_meals)
    
    with c1:
        st.metric("总消费金额", f"¥ {total_spent:.2f}")
    with c2:
        st.metric("记录总餐数", f"{total_meals} 餐")
    with c3:
        if total_meals > 0:
            st.metric("平均每餐价格", f"¥ {total_spent/total_meals:.2f}")

    # 红黑榜
    st.subheader("🏆 我的美食红黑榜（标记为“好吃”的餐品）")
    delicious = df_meals[df_meals["是否好吃"] == "好吃"]
    if not delicious.empty:
        st.dataframe(delicious[["日期", "餐别", "用餐地点", "菜品种类", "价格(元)"]], use_container_width=True)
    else:
        st.info("快去表格中给吃过的饭菜打上“好吃”标记吧！")
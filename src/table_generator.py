import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from ibm_watson_machine_learning.foundation_models import Model

from core import config
from .prompt_store import generate_ner_prompt

project_id = config.project_id
tax_item_col = config.tax_item_col
year_col = config.year_col

model_id = config.table_extraction_model_name
parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 250,
    "min_new_tokens": 1,
    "stop_sequences": ["\n"],
    "repetition_penalty": 1
}

model = Model(
	model_id = model_id,
	params = parameters,
	credentials = config.creds,
	project_id = project_id,
	)

def pivot_table_function(df, tax_item_col, year_col, value_col, item, year):
    filtered_df =  df[(df[tax_item_col].isin(item)) & (df[year_col].isin(year))]
    print("============Filtered Df==============")
    print(filtered_df)
#     return filtered_df.set_index(year_col).T
    filtered_df[year_col] = filtered_df[year_col].astype(str)
    filtered_df[value_col] = pd.to_numeric(filtered_df[value_col])
    pivot_table = filtered_df.pivot(columns=tax_item_col, index=year_col, values = value_col).rename_axis(None, axis=1).fillna(0)
    columns = pivot_table.columns.to_list()
    if "總計" in columns:
        print("Reordering columns")
        columns.append(columns.pop(columns.index('總計')))
        print(columns)
        return filtered_df, pivot_table.reindex(columns, axis=1)
    else:
    # try:
    #     column_order = ["總計", "關稅", "營利事業所得稅", "綜合所得稅", "遺產稅", "贈與稅", "貨物稅", "證券交易稅", "菸酒稅", "營業稅", "地價稅", "土地增值稅", "房屋稅", "使用牌照稅", "契稅", "印花稅", "娛樂"]
    #     pivot_table = filtered_df.pivot(columns=tax_item_col, index=year_col, values = value_col).rename_axis(None, axis=1).fillna(0)
    # except:
    #     pivot_table = filtered_df.pivot(columns=tax_item_col, index=year_col, values = "收入").rename_axis(None, axis=1).fillna(0)
        return filtered_df, pivot_table


def get_entities(question):
    prompt_input = generate_ner_prompt(question)
    return model.generate_text(prompt=prompt_input)

def post_process_entities(response_from_llm):
    print(response_from_llm)
    raw_dict = eval(response_from_llm)
    try:
        tax_item = raw_dict["稅收"]
        years = raw_dict["年份"]
        value_col = raw_dict["數據"][0]
    except:
        tax_item = ["總計"]
        years = [111]
        value_col = "收入"

    return tax_item, years, value_col

def get_table(df, question):
    llm_response = get_entities(question)
    tax_items, years, value_col = post_process_entities(llm_response)
    if not value_col is None:
        return pivot_table_function(df, tax_item_col, year_col, value_col, tax_items, years)
    else:
        return None
    
def chart_generator(pivot_table, question):
    if "經濟成長率" in question:
    #    from matplotlib import font_manager
    #    fontP = font_manager.FontProperties()
    #    fontP.set_family('SimHei')
       matplotlib.rcParams['font.family'] = ['Heiti TC']
       matplotlib.rcParams['axes.unicode_minus'] = False
       pivot_table.plot(kind="line",marker="o")
       plt.ylabel("百分比（%)")

       for col in pivot_table.columns:
           for x, y in enumerate(pivot_table[col]):
               plt.text(x+0.05, y+0.05, f'{y:.1f}%', color='black', ha='left', va='bottom', fontsize=8, rotation=45)
    #    plt.legend(prop=fontP)
       return plt.savefig(config.plot_name)






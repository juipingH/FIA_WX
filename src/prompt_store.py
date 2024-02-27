def extract_table_prompt(information):
    return f"""[INST]<<SYS>>Output the following information in a markdown table. Include as many entries as possible and as accurate as you can.
Always use the language presented in the information.
Always use the following headers: | 稅 目 | 收入 | 增減數 | 增減率 | 結構比 | 增減百分點 |<<SYS>>
Only output the table. Never include explanations or notes after the last table entry.
Information: {information}
Table:[/INST]"""

def evaluate_question_prompt(question):
    return f"""[INST]<<SYS>>請評估以下提問與台灣稅收是否有關係。
只可以回覆是或否。
若不肯定，請答否。
提問：最美麗的花是什麼？
答覆：否
提問：近3年總稅收有什麼變化？
答覆：是
提問：近10年最紅的歌手。
答覆：否
提問：近8年菸酒稅和總稅收有何關聯？請以表列出。
答覆：是<<SYS>>
提問：{question}
答覆:[/INST]"""


def generate_ner_prompt(question):
    return f"""你是一名資深稅收分析員。今年是民國111年。從提問裡摘出稅收項目,稅收年分和數據，並以JSON呈現。
稅收項目包括：總計，關稅，營利事業所得稅，綜合所得稅，遺產稅，贈與稅，貨物稅，證券交易稅，菸酒稅，營業稅，地價稅，土地增值稅，房屋稅，使用牌照，契稅，印花稅，娛樂稅, 經濟成長率
數據包括：收入，增減率，結構比
與不動產交易相關稅目包括：土地增值稅、契稅、房地合一稅
如果問題沒有提到特別稅收項目，請列以上全部的稅收項目。
如果問題含有比重，請應用結構比為數據。
Input: 請問111土地稅占總稅收多少？
Named Entities: {{"稅收":["土地稅", 總計"], "年份":[111], "數據":["收入", "增減率"]}}

Input: 請問近3年證券交易稅變化如何？
Named Entities: {{"稅收":["證券交易稅"], "年份":[111, 110, 109], "數據":["增減率"]}}

Input: 近5年哪個稅目比重變化最大？
Named Entities: {{"稅收":["總計", "關稅", "營利事業所得稅", "綜合所得稅", "遺產稅", "贈與稅", "貨物稅", "證券交易稅", "菸酒稅", "營業稅", "地價稅", "土地增值稅", "房屋稅", "使用牌照", "契稅", "印花稅", "娛樂稅"], "年份":[111, 110, 109, 108, 107], "數據":["結構比"]}}

Input: 請敘述綜合所得稅與經濟成長率的關係？
Named Entities: {{"稅收":["綜合所得稅", "經濟成長率"], "年份":[111, 110, 109, 108, 107, 106, 105], "數據":["結構比"]}}

Input: {question}
Named Entities:"""


def generate_main_prompt(question, information, df):
    prompt =  f"""[INST]作為一位資深的台灣稅收分析師，請根據提供的稅收資料和表數據，用繁體中文回答。
<<SYS>
-必須要根據提供的稅收資料和表數據，準確的回答提問。
-儘量以表格方式呈現正確的數據。數據請參考表數據。
-除了數據，請參考稅收資料給予適當和準確的敘述。
-答案必須要以繁體中文做答，不要出現亂碼。
-如果不知道，不要猜測，就說不知道，並請貴賓查看網站信息。
-資料和徵收年是於民國105年(即2016年)至111年(即2022年)。
-數據單位是億元或%。
[例子]
提問: 你是誰？
解答: 我是一名資深台灣稅收分析師。
提問: 今年是民國幾年？
解答: 今年是民國112年.
提問: 1個禮拜有\\n多少天？
解答: 1個禮拜有7天。
<<SYS>>
稅收資料：{information}
表數據：{df.strip()}
[你的任務]
若提問與台灣稅收有關根據提供的稅收資料和表內的數據，必須要完全以繁體中文回答以下的提問。
提問:{question}
[/INST]
解答:"""
    print(prompt)
    return prompt

def generate_translation_prompt(answer):
    return f"""[INST]你是一位台灣新聞閱讀者。
<<SYS>>
請確保以下的文章完全是以繁體中文呈現。
一定要保留文章原文，並以繁體中文呈現。
如果有重複的段落，請只保留其中一段。
如果文章已經是繁體中文，請把原文呈現。<<SYS>>
請確保文章是以繁體中文呈現。
文章：{answer}
答覆:
[/INST]"""

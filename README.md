# 赛题介绍
[比赛地址](http://www.aiwin.org.cn/competitions/86#learn_the_details)

网页自动导航任务：根据用户输入的自然语言指令，在网页中规划（输出）一条动作序列，自动化完成指令中包含的用户意图

根据指令意图与网页（网页中可能包含公告、提示等异常弹窗）功能的不同，分有不同的任务，常见的任务包括：检索、下载、收藏 、评论等。本赛题仅涉及信息检索任务，涉及的原子操作包括：输入框、下拉框、单选框、复选框、日期选择、导航树 、按钮。

![中国裁判文书网] (https://wenshu.court.gov.cn/static/images/help04.png ”中国裁判文书网“)

我们把填充表格的流程用cahtglm进行预测，训练样本构造样例如下：
```
{"prompt": "用户的需求是[请查询一下全新好2022年2月10日到2023年1月1日期间的公告]。请问在网页上执行什么操作？", "response": "用户使用的网站是[巨潮资讯]。在网页上[点击]名称为[查询]的[按钮];[点击]名称为[港股]的[按钮];[输入]值[全新好]进名称为[标题关键字]的[输入框];[输入]值[2022-02-10]进名称为[开始日期]的[日期输入框];[输入]值[2023-01-01]进名称为[结束日期]的[日期输入框];[点击]名称为[确认]的[按钮];", "history": []}
{"prompt": "用户的需求是[查询按时间排序的2017年的西班牙地区的电影]。请问在网页上执行什么操作？", "response": "用户使用的网站是[猫眼电影]。在网页上[点击]名称为[检索]的[按钮];[点击]名称为[西班牙]的[单选框];[点击]名称为[2017]的[单选框];[点击]名称为[按时间排序]的[单选框];", "history": []}

```

根据response生成最终提交的样式

![提交样例](http://cdn.aiwin.org.cn/def/2023S_T1_3.png ”提交样例“)

"instruction"为任务指令的自然语言形式，"key-value"为任务指令的槽位形式，key为操作节点的文本描述，dom_type为操作节点类别（DOM元素类别见“奖励函数（可选）”一节说明），value为操作值，action为动作。

- description.ipynb


# 训练数据下载

[链接](https://pan.baidu.com/s/1R9mD8oi_dDS_6ShAlAygwg?pwd=gw8d) （提取码: gw8d）

# 数据补充

- generate_data.ipynb


# 模型训练

生成训练数据集后 具体的训练代码请查看[chatglm ptuning](https://github.com/THUDM/ChatGLM2-6B/tree/main/ptuning)

# 预测

- predict.py

# prompt work

我们尝试给模型更多的提示信息，构造每个网站中所有模版的更全信息，作为提示

- future_work.ipynb

## 在此感谢打比赛的小伙伴
- @MrYXJ
- @苏维埃计算机
- @我不吃芒果
- @鼠鼠

[![Star History Chart](https://api.star-history.com/svg?repos=guodongxiaren/README&type=Date)](https://star-history.com/#dawoshi/2023AIWIN_Competition/edit/master/README.md&Date)

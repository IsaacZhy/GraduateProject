# 美国总统选举预测方法研究

## 项目介绍

本项目包括课题中的所有运行代码及数据，包括收集的演讲文本数据、使用数据集微调两个模型、生成柱状图数据、产生训练集以及搭建多层感知机模型。

## 运行环境

本项目编写时的运行环境为：

```python
datasets                      2.8.0
d2l                           0.17.6
evaluate                      0.4.0
huggingface-hub               0.11.1
matplotlib                    3.5.1
numpy                         1.21.5
notebook                      6.4.12
pandas                        1.2.4
torch                         1.13.1
transformers                  4.25.1
```

如需安装以上运行环境，请运行以下命令行指令：

```python
pip intall datasets==2.8.0
pip intall d2l==0.17.6
pip intall evaluate==0.4.0
pip intall huggingface-hub==0.11.1
pip intall matplotlib==3.5.1
pip intall numpy==1.21.5
pip intall notebook==6.4.12
pip intall pandas==1.2.4
pip intall torch==1.13.1
pip intall transformers==4.25.1
```

或使用``pip install --no-cache-dir -r requirements.txt``来安装所有项目编写时的运行环境。

## 使用说明

### 模型微调并绘图

包括用于模型微调的两个notebook文件，``Finetune on BERT.ipynb``以及``Finetune on RoBERTa.ipynb``，在运行前需要注意更改运行路径、预训练模型保存路径和微调后模型保存路径。其路径参数在文件中如下设置：

```python
working_directory = "C:/Users/m1500/Documents/Notebook/" #更改为目前工作目录或使用os.getcwd()来替代
pt_save_directory = "C:/Users/m1500/Documents/Notebook/bert-base-goemotions/" #更改为预训练模型的保存目录或同上替代
model_save_directory = "C:/Users/m1500/Documents/Notebook/pretrained_model/bert-base-goemotions/" #更改为微调模型的保存路径或同上替代
```

更改完成后即可顺序运行。本项目的预训练模型从Hugging Face上加载，微调模型在训练完成后上传至Hugging Face。如果不想重新进行微调，在进行后续处理文本前，直接从Hugging Face加载微调后的模型即可。要加载微调后的BERT和RoBERTa，将``model_name``后面的参数分别更改为``IsaacZhy/bert-base-goemotions``和``IsaacZhy/roberta-large-goemotions``，执行以下代码即可直接加载。

```python
pt_model = AutoModelForSequenceClassification.from_pretrained(
    model_name = model_name,
    problem_type = "multi_label_classification",
    num_labels = num_labels,
    id2label=id2label,
    label2id=label2id
    )
```

在两个文件后半部分均包含绘制柱状图的代码，加载模型后直接运行即可。注意演讲文本数据data文件夹应该放在工作目录下。

## 数据集生成

代码在``resultGenerator.ipynb``内，和上一步骤相同，需要加载微调后的模型。执行时同样需要data目录位于工作目录下。文件内默认时从Hugging Face加载，如果想要使用重新微调的模型，可以更改模型路径或将此部分代码复制到``Finetune on BERT.ipynb``或``Finetune on RoBERTa.ipynb``中完成微调（即train模块）之后的步骤处。

执行完毕后会在data目录下生成``result.csv``数据集文件，作为搭建多层感知机的数据集。

## 搭建并训练多层感知机

代码包含在``MLP.ipynb``内，需要数据集``result.csv``在同一工作目录下。训练过程实时进行绘图，可选择不同的超参数进行训练尝试。

# 汉语逆向最大分词算法（Reverse Maximum Match，RMM）

## 1.算法主要思想

从字符串的反方向出发，先截取后i个字符，与词典库中的词语进行对比。若比对**不成功**，则截取后i-1个字符进行对比，依次类推，直到仅剩第一个字符，自动进行截取，此次截取结束；若对比**成功**，则将该词语记录下来，并从句子中截取下来。直至句子全部被拆分为词语，以数组进行存储。

![iShot_2024-10-31_19.29.15](https://raw.githubusercontent.com/1910853272/image/master/img/202410311932776.png)

## 2.代码实现

```python
def split_words_reverse(line, corpus_set):
    '''
    反向最大匹配分词

    Parameters
    ----------
    line: str, 需要分词的中文字符串
    corpus_set: set, 词库

    Returns
    ----------
    str: 分词结果
    '''
    n_line = len(line)
    start, end = 0, n_line
    result = []
    while end > 0:
        if (end - 0) == 1:
            # 剩余一个字符时，直接加入结果
            result.append(line[start:end])
            return '/'.join(reversed(result)) + '/'
        current_word = line[start:end]
        if current_word in corpus_set:
            # 如果词语在词库中，加入结果并更新指针
            result.append(current_word)
            end = start
            start = 0
            continue
        else:
            if len(current_word) == 1:
                # 如果是单个字符，将其加入词库并加入结果
                corpus_set.add(current_word)
                result.append(current_word)
                end = start
                start = 0
                continue
            start += 1  # 缩短匹配长度
            continue
        end -= 1
    return '/'.join(reversed(result)) + '/'
```

## 3.输出结果

![precision_recall_histograms](https://raw.githubusercontent.com/1910853272/image/master/img/202411112142024.png)

![2024-11-11_214138](https://raw.githubusercontent.com/1910853272/image/master/img/202411112142633.png)


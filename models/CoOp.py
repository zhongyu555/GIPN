import os.path as osp

import torch
import torch.nn as nn

from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from open_clip.tokenizer import SimpleTokenizer,tokenize


class TextEncoder(nn.Module):
    def __init__(self, clip_model):

        super().__init__()

        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection


    def forward(self, prompts, tokenized_prompts):
        # 这里prompts [11,77,768], tokenized_prompts [11,77]


        x = prompts + self.positional_embedding  # self.positional_embedding [77, 768]
        x = x.permute(1, 0, 2)  # NLD -> LND [11,77,768] --> [77,11,768]
        x,_,_ = self.transformer(x)  # 这步得到的结果x维度仍然为[11,77,768], 维度不变
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x) # [11,77,768]
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        # print(tokenized_prompts.argmax(dim=-1))  tensor([11, 11, 11, 12, 11, 14, 12, 12, 13, 14], device='cuda:7')

        # self.text_projection维度是[768,768]
        #  x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] 是class_token， 维度为[11, 768]
        # 结果x维度是[11, 768]是每一类的class_token
        # tokenized_prompts.argmax(dim=-1)会得到每个句子结束标志49407对应位置的索引，因为49407是最大的值
        '''
        torch.arange(x.shape[0]) 会生成一个长度为 N 的 1D 张量，形如：[0,1,2,…,N−1]   用途：用于在每个类别维度上索引每个样本
        作用：找到每个 Prompt 的类别 Token 在序列中的位置索引, tokenized_prompts 是一个 2D 张量，形状为 [N,L]=[11,77]，表示 Batch 中每个 Prompt 的 Token 序列。
        tokenized_prompts.argmax(dim=-1) 找到每个序列中类别 Token 的位置索引：通常，类别 Token 是序列中最大值对应的位置,输出是一个 1D 张量维度为[11],即每个类别 Token 的位置集合列表
        '''

        '''
        为什么需要投影 @ self.text_projection？
        在 Transformer 或多模态模型中，嵌入特征（如文本特征）虽然已经有了固定的维度（768），但这些特征通常是从不同来源（如不同的层或网络结构）中提取的。这些特征可能有以下问题：
        1 特征的语义空间不同
        原始特征虽然具有相同的维度，但它们的语义空间可能不同，直接使用会导致计算不准确。
        投影的作用是对特征进行重新编码，使它们与目标特征空间对齐。
        2 提取高层次语义信息
        原始特征可能包含了过多的低层次信息，而投影矩阵可以通过训练学习过滤掉无关的噪声，保留任务相关的高层次特征。
        3 学习任务相关的特征表示
        投影矩阵中的参数是可训练的，模型会根据任务的损失函数（如对比损失）学习到更适合特定任务的特征表示
        '''
        return x
        # x维度为[11,768]



class PromptLearner(nn.Module):
    def __init__(self,
                 prompts,  # # eg. {"normal":[normal, good, healthy...], "abnormal":[abnormal, positive, disease]}
                 n_ctx, # prompt max len
                 CSC, # True or False multi prompt
                 class_token_position, # cls position
                 clip_model):

        super().__init__()

        ctx_dim = clip_model.ln_final.weight.shape[0] # ctx_dim=768

        self.ctx={}

        for cls in prompts:    # 这里的cls的值只有’normal', ‘abnromal'两种;   prompts, eg. {"normal":[normal, good, healthy...], "abnormal":[abnormal, positive, disease]}
            for position in class_token_position:  # 可以生成多位置的，但是这里目前只使用end
                if CSC:
                    ctx_vectors = torch.empty(len(prompts[cls]), n_ctx, ctx_dim).to(clip_model.device)  # [prompts[cls]类别数，n_ctx=8，ctx_dim]
                    # eg. cls='normal' , 那么ctx_vectors=[10, 8,768]，可学习的嵌入
                    # eg. cls='abnormal' , 那么ctx_vectors=[11, 8,768]，可学习的嵌入
                else:
                    ctx_vectors = torch.empty(n_ctx, ctx_dim).to(clip_model.device)   # # [n_ctx=8，ctx_dim]
                nn.init.normal_(ctx_vectors, std=0.02)
                self.ctx['{}_{}'.format(cls,position)]=nn.Parameter(ctx_vectors,requires_grad=True)   # 使用 Python 的字符串格式化方法 format，将类别和位置组合成一个唯一的字符串键

        self.ctx = nn.ParameterDict(self.ctx)  # to be optimized, 可学习的嵌入
        #  self.ctx 为Dict:2, key='nromal_end', value维度为[10,8,768];  key='abnromal_end', value维度为[11,8,768]
        """
        为什么要用 nn.ParameterDict？
        普通字典（如 self.ctx）不能直接被 PyTorch 优化器识别，即使它存储的是 nn.Parameter。
        使用 nn.ParameterDict 可以让 PyTorch 知道这些参数是模型的一部分，从而在训练过程中自动计算梯度并更新它们。
        """

        prompt_prefix = " ".join(["X"] * n_ctx)  # 生成一个固定长度n_ctx的前缀字符串str，填充内容为 "X", ed."X X X X X X X X"

        _tokenizer = SimpleTokenizer()  # 用于分词的工具。其功能类似于自然语言处理中的常用分词器, 它的核心作用是将输入文本（字符串）转换为 Token 序列（即一组整数索引或字符串片段）

        prompts_split={cls: [prompt.replace("_", " ")  for prompt in prompts[cls]] for cls in prompts}
        # prompts_split为Dict:2
        """
        示例：
        输入 prompts = {"class1": ["a_red_apple", "a_green_apple"]}。
        输出 prompts_split = {"class1": ["a red apple", "a green apple"]}
        eg. "in_good_health"  -->  "in good health"
        单是本论文中其实直接用的空格，没有用下划线连接
        """

        prompts_lens= {cls: [ len(_tokenizer.encode(prompt)) for prompt in prompts_split[cls]] for cls in prompts_split}  # 使用 _tokenizer.encode(prompt) 将每个 Prompt 转换为对应的 Token 列表，然后计算 Token 的数量
        #  prompts_lens为Dict:2, {'abnormal':[1,1,3,1,...], 'normal':[1,1,1,2,...]}
        #  这里直接调用_tokenizer.encode(prompt)进行编码，则没有加上开始和终止符号，所以len(_tokenizer.encode(prompt))=len('X X X X X X X X normal.')=10
        """
        示例：
        如果 prompts_split = {"class1": ["a red apple", "a green apple"]}：
        prompts_lens = {"class1": [3, 3]}
        eg. 本论文中设'normal'=['normal','healthy','in good healthy',...], 则 'normal':[1,1,3,...]
        """

        prompts_learnable_tokens = {cls:[prompt_prefix + " " + prompt + "." for prompt in prompts_split[cls]] for cls in prompts_split}
        # prompts_learnable_tokens为Dict:2, key='normal'对应value为一个{list:10},这list里面的10个元素分别字符串str eg. 'X X X X X X X X normal.'
        """
        示例：
        如果 prompt_prefix = "X X X X" 且 prompts_split = {"class1": ["a red apple", "a green apple"]}：
        prompts_learnable_tokens = {"class1": ["X X X X a red apple.", "X X X X a green apple."]}。
        """

        tokenized_prompts = {cls:torch.cat([tokenize(prompt) for prompt in prompts_learnable_tokens[cls]]).to(clip_model.device) for cls in prompts_learnable_tokens}
        # tokenize为分词操作，并返回一个形状为 [number of input strings, context_length] 的二维张量，填充或截断到指定的上下文长度 context_length。每个分词结果都会被起始标记（<start_of_text>）和结束标记（<end_of_text>）包裹
        # tokenize操作里面，调用了_tokenizer.encode(text) 将文本分词 ： all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
        # 故len(_tokenizer.encode(prompt))+2 = len(tokenize(prompt)),因为多起始标记（<start_of_text>）和结束标记（<end_of_text>）

        # tokenized_prompts为{dict:2}, key='normal'对应value是维度为[10,77]的张量； key='abnormal'对应value是维度为[11,77]的张量
        # 使用 tokenize() 将每个 Prompt 转换为模型可用的 Token，并将结果合并为一个张量, 每个类别的维度为[num_prompts, len_prompt]

        with torch.no_grad():
            embeddings = {cls:clip_model.token_embedding(tokenized_prompts[cls])  for cls in tokenized_prompts}  # 使用嵌入层生成 Prompt 的嵌入
            # embeddings为dict:2, key='normal'对应value是维度为[10,77,768]的张量, key='abnormal'对应value是维度为[11,77,768]的张量
        """
        使用 CLIP 模型的嵌入层，将 Token 序列转换为嵌入表示。
        输入是 Token 张量，形状为：[num_prompts,len_prompt]
        输出是嵌入张量，形状为：[num_prompts,len_prompt,embed_dim],  embed_dim 是嵌入向量的维度。
        """

        self.register_embeddings={}

        for cls in embeddings:  # Prompt 的嵌入前缀和嵌入后缀是不可学习的固定部分，起始和终止符号，还有后面填充的占位符，它们从嵌入中提取并单独保存
            self.register_embeddings['{}_token_prefix'.format(cls)]=embeddings[cls][:, :1, :]
            # 对于cls=’normal‘, self.register_embeddings['normal_token_prefix']为[10,1,768]张量

            self.register_embeddings['{}_token_suffix'.format(cls)]=embeddings[cls][:, 1 + n_ctx :, :]
            # 对于cls=’normal‘, self.register_embeddings['normal_token_suffix']为[10,68,768]张量
            # 包括class和.也算在后缀里面，终止符号，还有后面填充的占位符

        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts # 使用 tokenize() 将每个 Prompt 转换为模型可用的 Token，并将结果合并为一个张量
        self.prompts_lens = prompts_lens
        self.class_token_position = class_token_position


    def forward(self):
        cls_prompts={}

        for cls in self.tokenized_prompts:  # self.tokenized_prompts为与词汇表对应的编码

            prefix =  self.register_embeddings['{}_token_prefix'.format(cls)]  # 对于cls=’normal'[10,1,768]，这个是嵌入的前缀部分，是不可学习的，一般表示起始符号
            suffix =  self.register_embeddings['{}_token_suffix'.format(cls)]   # 对于cls=’normal'[10,68,768]这个是嵌入的后缀部分，是不可学习的，一般终止符号和占位符

            cls_prompts[cls]=[]

            for position in self.class_token_position:

                ctx = self.ctx['{}_{}'.format(cls,position)]  # self.ctx['normal-end'] 维度为[10,8,768], 这是我们需要的可学习的嵌入
                if ctx.dim() == 2:
                    ctx = ctx.unsqueeze(0).expand(len(self.prompts_lens[cls]), -1, -1)

                if position == "end":
                    prompts = torch.cat(
                        [
                            prefix,  # (n_cls, 1, dim) [10,1,768]
                            ctx,     # (n_cls, n_ctx, dim)  可学习的嵌入 eg. 对于‘normal':[10, 8, 768], 对于‘abnormal':[11, 8, 768]
                            suffix,  # (n_cls, *, dim) [10,68,768]
                        ],
                        dim=1,
                    )    # 拼接之后的prompts维度为[10,77,768]

                elif position == "middle":

                    half_n_ctx = self.n_ctx // 2
                    prompts = []

                    for i in range(len(self.prompts_lens[cls])):
                        p_len = self.prompts_lens[cls][i]

                        prefix_i = prefix[i : i + 1, :, :]
                        class_i = suffix[i : i + 1, :p_len, :]
                        suffix_i = suffix[i : i + 1, p_len:, :]
                        ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                        ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]

                        prompt = torch.cat(
                            [
                                prefix_i,     # (1, 1, dim)
                                ctx_i_half1,  # (1, n_ctx//2, dim)
                                class_i,      # (1, name_len, dim)
                                ctx_i_half2,  # (1, n_ctx//2, dim)
                                suffix_i,     # (1, *, dim)
                            ],
                            dim=1,
                        )
                        prompts.append(prompt)
                    prompts = torch.cat(prompts, dim=0)

                else :
                    assert position == "front"
                    prompts = []

                    for i in range(len(self.prompts_lens[cls])):
                        p_len = self.prompts_lens[cls][i]

                        prefix_i = prefix[i : i + 1, :, :]
                        class_i = suffix[i : i + 1, :p_len, :]
                        suffix_i = suffix[i : i + 1, p_len:, :]
                        ctx_i = ctx[i : i + 1, :, :]
                        prompt = torch.cat(
                            [
                                prefix_i,  # (1, 1, dim)
                                class_i,   # (1, name_len, dim)
                                ctx_i,     # (1, n_ctx, dim)
                                suffix_i,  # (1, *, dim)
                            ],
                            dim=1,
                        )
                        prompts.append(prompt)

                    prompts = torch.cat(prompts, dim=0)

                cls_prompts[cls].append(prompts)  # 将当前类别 cls 不同位置[end, middle, ...]的 Prompt 拼接在一起
            cls_prompts[cls]=torch.cat(cls_prompts[cls],dim=0)
        return cls_prompts
        # dict:2, 'normal'= [10,77,768], 'abnormal' = [11,77,768]
        # 返回的是完整的prompts的嵌入，可以直接送入文本编码器


class PromptMaker(nn.Module):
    """
    1. 输入 prompts 和 clip_model。
    2. 使用 PromptLearner 生成 prompt。
    3. 使用 TextEncoder 对 prompt 编码为文本特征。
    4. 支持多类 token 位置（end, middle, front）。
    5. 归一化和堆叠特征，返回类别文本嵌入。

    """

    def __init__(self,
                 prompts,
                 clip_model,
                 n_ctx: int=8,  # prompt max len
                 CSC: bool= True,  # True or False multi prompt  表示使用类别不同的可学习提示，即每个类别有它对应的可学习提示
                 class_token_position: list=['end'],  # cls position
                 ):

        super().__init__()
        assert 'normal' in prompts and 'abnormal' in prompts

        for position in class_token_position:
            assert  position in ['end','middle','front']

        self.prompt_learner = PromptLearner(prompts, n_ctx, CSC, class_token_position, clip_model)
        """
        PromptLearner的前向forward返回的值
        # [
        #    prefix,  # (n_cls, 1, dim)  已经是embedding ，感觉是起始符号
        #    ctx,     # (n_cls, n_ctx, dim)  可学习跟新的空张量[prompts类别数，n_ctx=8，ctx_dim]
        #    suffix,  # (n_cls, *, dim)   已经是embedding，对应的是normal、abnormal、或者说是 a normal apple编码之后的embedding嵌入
            ]
        # 
        """
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        #  tokenized_prompts为{dict:2}, key='normal'对应value是维度为[10,77]的张量； key='abnormal'对应value是维度为[11,77]的张量
        # tokenized_prompts是离散的词汇向量，对应着词汇在词汇表里面的编码。 使用 tokenize() 将每个 Prompt 转换为模型可用的 Token，并将结果合并为一个张量

        self.class_token_position = class_token_position
        self.text_encoder = TextEncoder(clip_model)  # 提取每个 Prompt 中class_token 的嵌入向量,并将嵌入向量通过线性变换投影到最终的特征空间

    def forward(self):
        prompts = self.prompt_learner()
        # dict:2, 'normal'= [10,77,768], 'abnormal' = [11,77,768]
        # prompts是完整的prompts的嵌入，可以直接送入文本编码器

        # prompts 为字典{dict:2} 然后里面有两个key分别是’normal,'abnormal',
        # key=’normal对应的value是一个维度为[10,77,768]的张量  ,这里的‘10’应该是指正常的类别有10类（eg. normal, healthy, clear...）
        # key=’abnormal对应的value是一个维度为[11,77,768]的张量 ,这里的‘11’应该是指异常的类别有11类（eg. abnormal, unhealthy, unclear...）
        """
        prompts
         [
            prefix,  # (n_cls, 1, dim)  已经是embedding ，感觉是起始符号
            ctx,     # (n_cls, n_ctx, dim)  可学习跟新的空张量[prompts类别数，n_ctx=8，ctx_dim]
            suffix,  # (n_cls, *, dim)   已经是embedding，对应的是normal、abnormal、或者说是 a normal apple编码之后的embedding嵌入
            ]
         
        """
        tokenized_prompts = self.tokenized_prompts  # tokenized_prompts是离散的词汇向量，对应着词汇在词汇表里面的编码
        # tokenized_prompts 为字典{dict:2} 然后里面有两个key分别是’normal,'abnormal',
        # key=’normal对应的value是一个维度为[10,77]的张量
        # key=’abnormal对应的value是一个维度为[11,77]的张量

        text_features=[]

        for cls in prompts:  # 对字典prompts的两个元素‘normal','abnormal' 进行遍历
            class_embedding = self.text_encoder(prompts[cls], tokenized_prompts[cls].repeat(len(self.class_token_position),1))  # todo 这里可以再仔细了解一下
            # 结果class_embedding维度是[11, 768]是每一类的class_token

            # 这里传入的tokenized_prompts[cls].repeat(len(self.class_token_position),1)， 作用是找到每个 Prompt 的类别 Token 在序列中的位置索引
            # repeat(len(self.class_token_position),1)沿第 0 维重复 len(self.class_token_position) 次，

            class_embedding = class_embedding.mean(dim=0)  # 这里操作之后 class_embedding维度为(768,)张量， 对的所有10 or 11 个类别向量（每个向量维度为 768）按元素求平均值, 使用10个768求平均获得1平均值768
            class_embedding = class_embedding / class_embedding.norm()
            # 对 class_embedding 进行 归一化处理，将其转换为单位向量（L2 范数为 1）
            # 这里文本已经归一化过了，故后面在与图像求余弦相似度时，只需要计算点积即可  # todo text 归一化

            text_features.append(class_embedding)
        text_features = torch.stack(text_features, dim=1)
        return text_features  # [768,2] 为normal和abnormal各自对应的所有类别的平均文本提示，
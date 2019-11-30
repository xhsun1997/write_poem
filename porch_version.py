import torch
import numpy as np
from torchnet import meter
import tqdm
path='./tang.npz'
datas=np.load(path)
data,word2id,id2word=datas['data'],datas['word2ix'].item(),datas['ix2word'].item()

assert type(word2id)==type(id2word)==dict
print(data.shape)
print("There is %d poems in the corpus,and each poem in length of %d " % (data.shape[0],data.shape[1]))

vocab_size=len(word2id)

class Poetry_Model(torch.nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim):
        super(Poetry_Model,self).__init__()
        self.hidden_dim=hidden_dim
        self.embedding_dim=embedding_dim
        self.embedding_layer=torch.nn.Embedding(num_embeddings=vocab_size,embedding_dim=embedding_dim)
        self.lstm_layer=torch.nn.LSTM(input_size=embedding_dim,hidden_size=self.hidden_dim,num_layers=2)
        self.linear_layer=torch.nn.Linear(in_features=self.hidden_dim,out_features=vocab_size)

    def forward(self,input_data,hidden_state=None):
        seq_len,batch_size=input_data.size()

        if hidden_state==None:
            h0=input_data.data.new(2,batch_size,self.hidden_dim).fill_(0).float()
            c0=input_data.data.new(2,batch_size,self.hidden_dim).fill_(0).float()
            h0=torch.autograd.Variable(h0)
            c0=torch.autograd.Variable(c0)
        else:
            h0,c0=hidden_state

        embedding_layer_out=self.embedding_layer(input_data)
        assert embedding_layer_out.size()==(seq_len,batch_size,self.embedding_dim)
        lstm_out,lstm_state=self.lstm_layer(embedding_layer_out,(h0,c0))
        assert lstm_out.size()==(seq_len,batch_size,self.hidden_dim)
        assert len(lstm_state)==2 and lstm_state[0].size()==lstm_state[1].shape==(2,batch_size,self.hidden_dim)
        linear_out=self.linear_layer(lstm_out)
        assert linear_out.shape==(seq_len,batch_size,vocab_size)
        output=linear_out.view(seq_len*batch_size,vocab_size)

        return output,lstm_state

class Config(object):
    lr = 1e-3
    weight_decay = 1e-4
    use_gpu = False
    epoch = 20
    batch_size = 128
    maxlen = 125 # 超过这个长度的之后字被丢弃，小于这个长度的在前面补空格
    plot_every = 20 # 每20个batch 可视化一次
    env='poetry' # visdom env
    max_gen_len = 200 # 生成诗歌最长长度
    prefix_words = '细雨鱼儿出,微风燕子斜。' # 不是诗歌的组成部分，用来控制生成诗歌的意境
    start_words='闲云潭影日悠悠' # 诗歌开始
    acrostic = False # 是否是藏头诗
    model_prefix = './log/tang_poem' # 模型保存路径

opt = Config()

dataloader=torch.utils.data.DataLoader(data,batch_size=opt.batch_size,shuffle=True,num_workers=1)

def generate(model,start_word,word2id,id2word,prefix_word=None):
    results=list(start_word)
    start_word_len=len(start_word)
    print(start_word)

    assert start_word_len==1
    input_=torch.Tensor([word2id['<START>']]).view(1,1).long()
    if Config.use_gpu:
        input_=input_.cuda()
    hidden=None
    if prefix_word:
        for word in prefix_word:
            output,state=model(input_,hidden)
            input_=input_.data.new([word2id[word]]).view(1,1)
            input_=torch.autograd.Variable(input_)
    #If prefix_word!=None we can then see the input_ is the last word of prefix_word and hidden has its own value
    #If prefix_word==None, we can see then input_ is the tensor of <START> and hidden=None
    for i in range(Config.max_gen_len):
        output,hidden=model(input_,hidden)
        if i<start_word_len:
            assert i==0
            w=results[i]
            w_=start_word[i]
            assert w==w_
            input_=input_.data.new([word2id[w]]).view(1,1)
            input_=torch.autograd.Variable(input_)
        else:
            assert i>=1
            top_index=output.data[0].topk(1)[1][0].item()
            w=id2word[top_index]
            results.append(w)
            input_=input_.data.new([top_index]).view(1,1)
        if w=='<EOP>':
            del results[-1]
            break
    return results




def train():
    model=Poetry_Model(vocab_size=vocab_size,embedding_dim=150,hidden_dim=256)
    optimizer=torch.optim.Adam(model.parameters(),lr=opt.lr)
    criterion=torch.nn.CrossEntropyLoss()
    if opt.use_gpu:
        model.cuda()
        criterion.cuda()
    loss_meter=meter.AverageValueMeter()
    batch_size=opt.batch_size
    maxlen=opt.maxlen
    for epoch in range(opt.epoch):
        loss_meter.reset()
        for i,data_ in tqdm.tqdm(enumerate(dataloader)):
            assert data_.shape==(batch_size,maxlen)
            data_=data_.long().transpose(1,0).contiguous()
            assert data_.size()==(maxlen,batch_size)
            if opt.use_gpu:
                data_=data_.cuda()
            optimizer.zero_grad()
            input_data=torch.autograd.Variable(data_[:-1,:])
            target=torch.autograd.Variable(data_[1:,:])
            assert input_data.size()==(maxlen-1,batch_size)==target.shape

            model_output,states=model(input_data)
            loss=criterion(input=model_output,target=target.view(-1))
            loss.backward()
            optimizer.step()
            loss_meter.add(loss.item())
            if (i)%opt.plot_every==0:
                print("loss value is ",loss_meter.mean)
                for word in list(u"春江花朝秋月夜"):
                    gen_poetry="".join(generate(model,word,word2id,id2word))
                    print(gen_poetry)

        torch.save(model.state_dict(),'%s_%s.pth'%(opt.model_prefix,epoch))

if __name__ == '__main__':
    train()













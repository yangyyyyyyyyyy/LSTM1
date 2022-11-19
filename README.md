# LSTM1
设计思路
①　遗忘门：选择不必要的信息并忘记。决定了t-1时刻的细胞状态C中的哪些信息将被遗忘。包含一个sigmoid的神经网络层。接收t时刻的输入信号xt和t-1时刻LSTM的上一个输出信号ht-1，这两个信号进行拼接以后共同输入到神经网络层中，然后输出信号 ，是一个  到之间的数值，并与  相乘来决定 中的哪些信息将被保留，哪些信息将被舍弃。

②　记忆门：决定新输入的信息xt和ht-1中哪些信息将被保留。包含两个       部分，一个sigmoid神经网络层（输入门）和一个tanh神经网络层
Sigmoid神经网络层：接收xt和ht-1作为输入，然后输出一个0到1之间的数值I来决定哪些信息需要被更新.
Tanh神经网络层：将输入的xt和ht-1整合，然后通过一个神经网络层来创建一个新的状态候选向量C_t，C_t 的值范围在-1到1之间
③　利用遗忘门以及记忆门更新细胞状态C， C = F * C + I * C_t，新的细胞  状态将继续传递到t+1时刻的LSTM网络中
④　输出门：将t-1时刻传递过来并经过了前面遗忘门与记忆门选择后的细胞状态C，与t-1时刻的输出信号ht-1和t时刻的输入信号xt整合到一起作为当前时刻的输出信号。输出门由一个sigmoid神经网络层与一个tanh函数组成。
def forward(self, X):
        X = self.C(X)
        X = X.transpose(0, 1)  # X : [n_step, batch_size, n_class]
        sample_size = X.size()[1]
        """do this LSTM forward"""
        """begin"""
        ##Complete your code with the hint: a^(t) = tanh(W_{ax}x^(t)+W_{aa}a^(t-1)+b_{a})  y^(t)=softmx(Wa^(t)+b)
        # 初始化隐藏层状态全0
        H = torch.zeros([sample_size, n_hidden]).to(device)
        C = torch.zeros([sample_size, n_hidden]).to(device)
        for x in X:
            F = self.sigmoid(self.W_xf(x) + self.W_hf(H) + self.b_f)#遗忘门的sigmoid神经网络
            I = self.sigmoid(self.W_xi(x) + self.W_hi(H) + self.b_i)#记忆门的sigmoid神经网络层
            C_t = self.tanh(self.W_xc(x) + self.W_hc(H) + self.b_c)#记忆门的tanh神经网络层
O = self.sigmoid(self.W_xo(x) + self.W_ho(H) + self.b_o)#输出门的sigmoid神经网络层
            C = F * C + I * C_t#更新细胞状态
            H = O * torch.tanh(C)#得到输出信号

        model_output = self.W_hq(H) + self.b_q
        """end"""
        return model_output

在模型构建的过程中，首先将隐藏层初始化，然后开始循环，利用遗忘层的sigmoid神经网络层和记忆门的sigmoid神经网络层以及tanh神经网络层，更新细胞状态（ F * C + I * C_t）

#include <iostream>
using namespace std;
#include <string>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "matrix.h"
#include <fstream>
#define MAX_LAYER_N 100

/**
网络参数W,b其中Z,A是为了求解残差方便
**/
struct parameters
{
    Matrix W;
    Matrix WT;
    Matrix b;
    Matrix Z;
    Matrix A;
    Matrix AT;
    Matrix D;
    parameters *next;
    parameters *pre;
};
parameters par;// 定义全局变量

/***
复合函数中对应目标函数对相应变量的偏导
**/
struct grad
{
    Matrix grad_W;
    Matrix grad_b;
    Matrix grad_Z;
    Matrix grad_A;
    Matrix V_dw;
    Matrix V_db;
    Matrix S_dw;
    Matrix S_db;
    Matrix V_dw_corrected;
    Matrix V_db_corrected;
    Matrix S_dw_corrected;
    Matrix S_db_corrected;
    grad *next;
    grad *pre;
};
grad gra;//定义全局变量

/**
神经网络超参数
**/
struct sup_parameters
{
    int layer_dims;//神经网络层数
    int layer_n[MAX_LAYER_N];//每层神经元个数
    string layer_active[MAX_LAYER_N];//每层激活函数
};
sup_parameters sup_par;//定义全局变量

/**
所有参数都定义为全局标量，结构体不需要在函数之间传递，
根据超参数初始化参数
**/
int init_parameters(Matrix X,const char *initialization)
{
    int k=0,i=0,j=0;
    double radom;
    int L=sup_par.layer_dims;//网络层数
    parameters *p=&par;//参数，结构体已定义并分配内存，结构体内矩阵未分配内存
    grad *g=&gra;//梯度，结构体已定义并分配内存，结构体内矩阵未分配内存
    /**
        随机初始化
    **/
    p->A.initMatrix(&(p->A),X.col,X.row);
    p->AT.initMatrix(&(p->AT),X.row,X.col);
    for(k=0; k<L-1; k++)
    {
        p->W.initMatrix(&p->W,sup_par.layer_n[k+1],sup_par.layer_n[k]);
        p->WT.initMatrix(&p->WT,sup_par.layer_n[k],sup_par.layer_n[k+1]);

        p->b.initMatrix(&p->b,sup_par.layer_n[k+1],1);
        p->Z.initMatrix(&p->Z,sup_par.layer_n[k+1],X.row);

        //用于dropout，这里初始化一次即可，后面当使用dropout时，D才会赋值，不使用则不赋值，且实际使用长度小于网络层数
        p->D.initMatrix(&p->D,p->A.col,p->A.row);

        for(i=0; i<p->W.col; i++)
        {
            for(j=0; j<p->W.row; j++)
            {
                if(initialization=="he")
                {
                    radom=(rand()%100)/100.0;
                    p->W.mat[i][j]=radom * sqrt(2.0/sup_par.layer_n[k]);//一种常用的参数初始化方法，参数初始化也有技巧
                }
                if(initialization=="random")
                {
                    radom=(rand()%100)/100.0;
                    p->W.mat[i][j]=radom;//一种常用的参数初始化方法，参数初始化也有技巧
                }
                if(initialization=="arxiv")
                {
                    radom=(rand()%100)/100.0;
                    p->W.mat[i][j]=radom/sqrt(sup_par.layer_n[k]);//一种常用的参数初始化方法，参数初始化也有技巧
                }
            }
        }
        p->next=new parameters();//下一层网络参数
        p->next->pre=p;
        p=p->next;

        g->grad_A.initMatrix(&(g->grad_A),sup_par.layer_n[L-k-1],X.row);
        g->grad_Z.initMatrix(&(g->grad_Z),sup_par.layer_n[L-k-1],X.row);
        g->grad_W.initMatrix(&(g->grad_W),sup_par.layer_n[L-k-1],sup_par.layer_n[L-k-2]);
        g->grad_b.initMatrix(&(g->grad_b),sup_par.layer_n[L-k-1],1);

        //用于momentum 和adam优化中用于保存前n次加权平均值
        g->V_dw.initMatrix(&g->V_dw,sup_par.layer_n[L-k-1],sup_par.layer_n[L-k-2]);
        g->V_db.initMatrix(&g->V_db,sup_par.layer_n[L-k-1],1);

        g->S_dw.initMatrix(&g->S_dw,sup_par.layer_n[L-k-1],sup_par.layer_n[L-k-2]);
        g->S_db.initMatrix(&g->S_db,sup_par.layer_n[L-k-1],1);


        //用于修正的momentum 和adam
        g->V_dw_corrected.initMatrix(&g->V_dw_corrected,sup_par.layer_n[L-k-1],sup_par.layer_n[L-k-2]);
        g->V_db_corrected.initMatrix(&g->V_db_corrected,sup_par.layer_n[L-k-1],1);

        g->S_dw_corrected.initMatrix(&g->S_dw_corrected,sup_par.layer_n[L-k-1],sup_par.layer_n[L-k-2]);
        g->S_db_corrected.initMatrix(&g->S_db_corrected,sup_par.layer_n[L-k-1],1);


        g->pre=new grad();//上一层网络参数梯度
        g->pre->next=g;
        g=g->pre;


        p->A.initMatrix(&(p->A),sup_par.layer_n[k+1],X.row);
        p->AT.initMatrix(&(p->AT),X.row,sup_par.layer_n[k+1]);
    }
    g->grad_A.initMatrix(&(g->grad_A),sup_par.layer_n[L-k-1],X.row);

    return 0;
}

void line_forward(parameters *p,double keep_prob)
{
    int i=0,j=0;
    if(keep_prob!=1)
    {
       for(i=0; i<p->D.col; i++)
        {
            for(j=0; j<p->D.row; j++)
            {
                p->D.mat[i][j]=(rand()%100)/100.0;

                if(p->D.mat[i][j]<keep_prob)
                    p->D.mat[i][j]=1.0/keep_prob; //这里已经扩充了keep_prob
                else
                    p->D.mat[i][j]=0;
            }
        }
        p->A.mult(&p->A,p->A,p->D);
    }
    p->Z.multsmatrix(&p->Z,p->W,p->A);
    for(i=0; i<p->Z.col; i++) //矩阵与向量的相加，class中未写
    {
        for(j=0; j<p->Z.row; j++)
        {
            p->Z.mat[i][j]+=p->b.mat[i][0];//这里可以把b也定义为等大小的矩阵，每行一样
        }
    }
}

void sigmoid_forward(parameters *p)
{
    int i,j;
    for(i=0; i<p->Z.col; i++)
    {
        for(j=0; j<p->Z.row; j++)
        {
            p->next->A.mat[i][j]=1.0/(1.0+exp(-p->Z.mat[i][j]));
        }
    }
}

void relu_forward(parameters *p)
{
    int i,j;
    for(i=0; i<p->Z.col; i++)
    {
        for(j=0; j<p->Z.row; j++)
        {
            if(p->Z.mat[i][j]>0)
            {
                p->next->A.mat[i][j]=p->Z.mat[i][j];
            }
            else
            {
                p->next->A.mat[i][j]=0;
            }
        }
    }
}

void line_active_forward(parameters *p,string active, double keep_prob)
{
    line_forward(p,keep_prob);
    if(active=="relu")
    {
        relu_forward(p);
    }
    if(active=="sigmoid")
    {
        sigmoid_forward(p);
    }
}

Matrix model_forward(Matrix X,double *keep_probs)
{
    int i=0;
    int L=sup_par.layer_dims;

    parameters *p=&par;
    p->A.copy(X,&p->A);
    for(i=0; i<L-1&&p->next!=NULL; i++)
    {
        line_active_forward(p,sup_par.layer_active[i+1],keep_probs[i]);
        p=p->next;
    }
    return p->A;
}
void sigmoid_backword(parameters *p,grad *g)
{
    int i=0,j=0;

    for(i=0; i<g->grad_A.col; i++)
    {
        for(j=0; j<g->grad_A.row; j++)
        {
            g->grad_Z.mat[i][j]=g->grad_A.mat[i][j]*p->A.mat[i][j]*(1-p->A.mat[i][j]);
        }
    }
}

void relu_backword(parameters *p,grad *g)
{
    int i=0,j=0;

    for(i=0; i<g->grad_Z.col; i++)
    {
        for(j=0; j<g->grad_Z.row; j++)
        {
            if(p->pre->Z.mat[i][j]>0)
            {
                g->grad_Z.mat[i][j]=g->grad_A.mat[i][j];
            }
            else
            {
                g->grad_Z.mat[i][j]=0;
            }
        }
    }
}

void line_backword(parameters *p,grad *g, double lambd, double keep_prob)
{
    int i,j;
    p->AT.transposematrix(p->A,&p->AT);
    g->grad_W.multsmatrix(&g->grad_W,g->grad_Z,p->AT);
    if(lambd!=0)
    {
        for(i=0; i<p->W.col; i++)
        {
            for(j=0; j<p->W.row; j++)
            {
                 g->grad_W.mat[i][j]+=(lambd * p->W.mat[i][j]);
            }
        }
    }
    for(i=0; i<g->grad_W.col; i++)
    {
        for(j=0; j<g->grad_W.row; j++)
        {
            g->grad_W.mat[i][j]/=g->grad_Z.row;
        }
    }
    for(i=0; i<g->grad_Z.col; i++)
    {
        g->grad_b.mat[i][0]=0;
        for(j=0; j<g->grad_Z.row; j++)
        {
            g->grad_b.mat[i][0]+=g->grad_Z.mat[i][j];
        }
        g->grad_b.mat[i][0]/=g->grad_Z.row;
    }
    p->WT.transposematrix(p->W,&p->WT);
    g->pre->grad_A.multsmatrix(&g->pre->grad_A,p->WT,g->grad_Z);

    if(keep_prob!=1)
    {
        //这里p指向的D与对应A的dropout层，而等于1的情况下，D是只有初始化，无关赋值，所以对应dropout关系是正确的
        //cout<<p->D.col<<"&"<<p->D.row<<endl;
        //cout<<g->pre->grad_A.col<<"&"<<g->pre->grad_A.row<<endl;

        g->pre->grad_A.mult(&g->pre->grad_A,g->pre->grad_A,p->D);//由于keep_prob扩充已经放到D上了

        //p->D.print(p->next->D);
        //cin>>i;
    }
}

void line_active_backword(parameters *p,grad *g,string active, double lambd, double keep_prob)
{
    if(active=="sigmoid")
    {
        sigmoid_backword(p,g);
        line_backword(p->pre,g,lambd,keep_prob);
    }

    if(active=="relu")
    {
        relu_backword(p,g);
        line_backword(p->pre,g,lambd,keep_prob);
    }
}
void model_backword(Matrix AL,Matrix Y,double lambd,double *keep_probs)
{
    int i=0;
    int L=sup_par.layer_dims;

    parameters *p=&par;
    while(p->next!=NULL)
    {
        p=p->next;
    }
    grad *g=&gra;

    for(i=0; i<Y.row; i++)
    {
        gra.grad_A.mat[0][i]=-(Y.mat[0][i]/AL.mat[0][i]-(1-Y.mat[0][i])/(1-AL.mat[0][i]));
    }
    for(i=L-1; i>0; i--)
    {
        line_active_backword(p,g,sup_par.layer_active[i],lambd,keep_probs[i]);
        g=g->pre;
        p=p->pre;
    }
}

double cost_cumpter(Matrix AL,Matrix Y,double lambd)
{
    int i=0,j=0;
    int n=Y.row;
    double loss=0;
    double loss_L2_regularization=0;
    if(lambd!=0)
    {
        parameters *p=&par;
        while(p!=NULL)
        {
            for(i=0;i<p->W.col;i++)
            {
                for(j=0;j<p->W.row;j++)
                {
                    loss_L2_regularization+=(lambd*p->W.mat[i][j]*p->W.mat[i][j]);
                }
            }
           p=p->next;
        }
        loss_L2_regularization/=n;
    }
    for(i=0; i<n; i++)
    {
        loss+=-(Y.mat[0][i]*log(AL.mat[0][i])+(1-Y.mat[0][i])*log(1-AL.mat[0][i]));
    }
    loss/=n;
    loss+=loss_L2_regularization;
    return loss;
}
int updata_parameters_with_gd(double learn_rateing, int t)
{
    int k=0,i=0,j=0;
    int L=sup_par.layer_dims;
    parameters *p=&par;
    grad *g=&gra;
    while(g->pre->pre!=NULL)//反向和前向组合参数的方式不同，前者A=sgn(WX+b),后者 AL=sgn(WA+b)
    {
        g=g->pre;
    }

    learn_rateing *= pow(0.999, t/1000);//指数衰减

    for(k=0; k<L-1&&p->next!=NULL&&g!=NULL; k++)
    {
        for(i=0; i<g->grad_W.col; i++)
        {
            g->grad_b.mat[i][0]*=-learn_rateing;
            for(j=0; j<g->grad_W.row; j++)
            {
                g->grad_W.mat[i][j]*=-learn_rateing;
            }
        }
        p->W.addmatrix(&p->W,p->W,g->grad_W);
        p->b.addmatrix(&p->b,p->b,g->grad_b);
        p=p->next;
        g=g->next;
    }
    return 0;
}

int updata_parameters_with_momentum(double learn_rateing, int t,double beta)
{
    int k=0,i=0,j=0;
    int L=sup_par.layer_dims;
    parameters *p=&par;
    grad *g=&gra;
    while(g->pre->pre!=NULL)//反向和前向组合参数的方式不同，前者A=sgn(WX+b),后者 AL=sgn(WA+b)
    {
        g=g->pre;
    }

    //learn_rateing *= pow(0.999, t/1000);//指数衰减

    for(k=0; k<L-1&&p->next!=NULL&&g!=NULL; k++)
    {
        for(i=0; i<g->grad_W.col; i++)
        {
            g->V_db.mat[i][0]=(beta * g->V_db.mat[i][0] + (1-beta) * g->grad_b.mat[i][0]);
            g->V_db_corrected.mat[i][0] = g->V_db.mat[i][0] / (1-pow(beta,t));//修正
            g->grad_b.mat[i][0]=(-learn_rateing) * g->V_db_corrected.mat[i][0];

            for(j=0; j<g->grad_W.row; j++)
            {
                g->V_dw.mat[i][j]=(beta * g->V_dw.mat[i][j] + (1-beta) * g->grad_W.mat[i][j]);
                g->V_dw_corrected.mat[i][j]=g->V_dw.mat[i][j] / (1-pow(beta,t));//修正
                g->grad_W.mat[i][j]=(-learn_rateing) * g->V_dw_corrected.mat[i][j];
            }
        }
        p->W.addmatrix(&p->W,p->W,g->grad_W);
        p->b.addmatrix(&p->b,p->b,g->grad_b);
        p=p->next;
        g=g->next;
    }
    return 0;
}

int updata_parameters_with_adam(double learn_rateing, int t, double beta1, double beta2, double epsilon)
{
    int k=0,i=0,j=0;
    int L=sup_par.layer_dims;
    parameters *p=&par;
    grad *g=&gra;
    while(g->pre->pre!=NULL)//反向和前向组合参数的方式不同，前者A=sgn(WX+b),后者 AL=sgn(WA+b)
    {
        g=g->pre;
    }

    learn_rateing *= pow(0.999, t/1000);//指数衰减

    for(k=0; k<L-1&&p->next!=NULL&&g!=NULL; k++)
    {
        for(i=0; i<g->grad_W.col; i++)
        {
            g->V_db.mat[i][0]=(beta1 * g->V_db.mat[i][0] + (1-beta1) * g->grad_b.mat[i][0]);
            g->V_db_corrected.mat[i][0] = g->V_db.mat[i][0] / (1-pow(beta1,t));//修正

            g->S_db.mat[i][0]=(beta2 * g->S_db.mat[i][0] + (1-beta2) * (g->grad_b.mat[i][0] * g->grad_b.mat[i][0]));
            g->S_db_corrected.mat[i][0] = g->S_db.mat[i][0] / (1-pow(beta2,t));//修正

            g->grad_b.mat[i][0]= (-learn_rateing) * g->V_db_corrected.mat[i][0] / sqrt(g->S_db_corrected.mat[i][0]);

            for(j=0; j<g->grad_W.row; j++)
            {
                g->V_dw.mat[i][j]=(beta1 * g->V_dw.mat[i][j] + (1-beta1) * g->grad_W.mat[i][j]);
                g->V_dw_corrected.mat[i][j]=g->V_dw.mat[i][j] / (1-pow(beta1,t));//修正

                g->S_dw.mat[i][j]=(beta2 * g->S_dw.mat[i][j] + (1-beta2) * (g->grad_W.mat[i][j] * g->grad_W.mat[i][j]));
                g->S_dw_corrected.mat[i][j]=g->S_dw.mat[i][j] / (1-pow(beta2,t));//修正

                g->grad_W.mat[i][j]= (-learn_rateing) * g->V_dw_corrected.mat[i][j] / sqrt(g->S_dw_corrected.mat[i][j]+epsilon) ;
            }
        }
        p->W.addmatrix(&p->W,p->W,g->grad_W);
        p->b.addmatrix(&p->b,p->b,g->grad_b);
        p=p->next;
        g=g->next;
    }
    return 0;
}

int updata_parameters(double learn_rateing, int t, const char *optimizer, double beta1, double beta2, double epsilon)
{

    if(optimizer=="gd")
       updata_parameters_with_gd(learn_rateing, t);
    else if(optimizer="momentum")
        updata_parameters_with_momentum(learn_rateing, t, beta1);
    else if(optimizer="adam")
        updata_parameters_with_adam(learn_rateing, t, beta1, beta2, epsilon);
    return 0;
}

int DNN(Matrix X,Matrix Y,const char *optimizer,double learn_rateing,const char *initialization, double lambd, double keep_prob, \
        int mini_batch_size,double beta1, double beta2, double epsilon, int iter, bool print_cost)
{
    /**
    初始化参数

    **/
    int i=0,j=0,k=0;
    int lay_dim=3;
    int lay_n[3]= {21,16,1};
    string lay_active[3]= {"relu","relu","sigmoid"};


    sup_par.layer_dims=lay_dim;
    for(i=0; i<lay_dim; i++)
    {
        sup_par.layer_n[i]=lay_n[i];
        sup_par.layer_active[i]=lay_active[i];
    }
    init_parameters(X,initialization);
    double loss;

    Matrix AL;
    AL.initMatrix(&AL,Y.col,Y.row);
    double *keep_probs;
    if(keep_prob==1)
    {
        keep_probs=new double [sup_par.layer_dims];
        for(k=0;k<sup_par.layer_dims;k++)
        {
            keep_probs[k]=1;
        }
    }
    else if (keep_prob<1)
    {
        keep_probs=new double [sup_par.layer_dims];
        for(k=0;k<sup_par.layer_dims;k++)
        {
            if(k==0 || k==sup_par.layer_dims-1)
            {
                keep_probs[k]=1;
            }
            else
            {
                keep_probs[k]=0.99;
            }
        }
    }
    for(i=0; i<iter; i++)
    {
        //cout<<"-----------forward------------"<<"i="<<i<<endl;
        AL=model_forward(X,keep_probs);
        //cout<<"-----------loss--------------"<<endl;
        loss=cost_cumpter(AL,Y,lambd);
        if(i%1000==0)
            cout<<"loss="<<loss<<endl;
        //cout<<"-----------backword-----------"<<endl;
        model_backword(AL,Y,lambd,keep_probs);

        //cout<<"-----------update--------------"<<endl;
        updata_parameters(learn_rateing,i+1,optimizer,beta1,beta2,epsilon);
    }
    return 0;
}
int predict(Matrix X,Matrix Y)
{
    int i,j,k;
    parameters *p;
    p=&par;
    p->A.copy(X,&p->A);
    Matrix AL;
    double *keep_probs=new double [sup_par.layer_dims];
    for(k=0;k<sup_par.layer_dims;k++)
    {
        keep_probs[k]=1;
    }
    AL=model_forward(X,keep_probs);
    for(i=0;i<Y.row;i++)
    {
        if(AL.mat[0][i]>0.5)
            AL.mat[0][i]=1;
        else
            AL.mat[0][i]=0;
    }
    double pre=0;
    for(i=0;i<Y.row;i++)
    {
        if((AL.mat[0][i]==1 && Y.mat[0][i]==1)||(AL.mat[0][i]==0 && Y.mat[0][i]==0))
            pre+=1;
    }
    pre/=Y.row;
    cout<<"pre="<<pre<<endl;
    return 0;
}
int main()
{
    /**
        加载数据
    **/
    dataToMatrix dataset;
    const char *data;
    data="datasets//test.txt";
    dataset.loadData(&dataset,data);

    /**
        将数据转换成矩阵形式

    **/
    Matrix train_data,train_dataT;
    train_data.loadMatrix(&train_data,dataset);
    //train_dataT.initMatrix(&train_dataT,train_data.row,train_data.col);
    //train_dataT.transposematrix(train_data,&train_dataT);

    /**
        生成输入输出矩阵
    **/
    Matrix X;
    Matrix Y;
    X.initMatrix(&X,train_data.col,train_data.row);
    Y.initMatrix(&Y,1,train_data.row);
    X.copy(train_data,&X);
    Y=Y.getOneCol(X,X.col);
    X.deleteOneCol(&X,X.col);


    /**
        归一化很重要
    **/
    int i=0,j=0;
    for(i=0;i<X.col;i++)
    {
        for(j=0;j<X.row;j++)
        {
            X.mat[i][j]/=255;
        }
    }

    const char *initialization="he";
    double learn_rateing=0.001;
    int iter=1000;
    double lambd=0.1;
    double keep_prob=0.5;
    bool print_cost=true;
    /**
        神经网络调用
    **/
    const char *optimizer="gd";
    int mini_batch_size=64;
    double beta1=0.9;
    double beta2=0.999;
    double epsilon=0.00000001;

    DNN(X,Y,optimizer="adam",learn_rateing=0.1,initialization="he",lambd=0.01,keep_prob = 1,mini_batch_size=64, \
        beta1=0.9, beta2=0.999, epsilon=0.00000001, iter=50000, print_cost=true);

    /**
        输出参数
    **/
    /*
    parameters *p;
    p=&par;
    while(p->next!=NULL)
    {
        cout<<"W======"<<endl;
        p->W.print(p->W);
        cout<<"D======="<<endl;
        p->D.print(p->D);
        p=p->next;
    }
    */

    predict(X,Y);
    return 0;
}

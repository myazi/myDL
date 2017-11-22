#include <iostream>
using namespace std;
#include <string>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "matrix.h"
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
    parameters *next;
    parameters *pre;
};
parameters par;

/***
复合函数中对应目标函数对相应变量的偏导
**/
struct grad
{
    Matrix grad_W;
    Matrix grad_b;
    Matrix grad_Z;
    Matrix grad_A;
    grad *next;
    grad *pre;
};
grad gra;

/**
神经网络超参数
**/
struct sup_parameters
{
    int layer_dims;//神经网络层数
    int layer_n[MAX_LAYER_N];//每层神经元个数
    string layer_active[MAX_LAYER_N];//每层激活函数
};
sup_parameters sup_par;

/**
所有参数都定义为全局标量，结构体不需要在函数之间传递，
根据超参数初始化参数
**/
int init_parameters(Matrix X)
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
    p->A.copy(X,&p->A);
    for(k=0; k<L-1; k++)
    {
        p->W.initMatrix(&(p->W),sup_par.layer_n[k+1],sup_par.layer_n[k],0);
        p->WT.initMatrix(&(p->WT),sup_par.layer_n[k],sup_par.layer_n[k+1]);

        p->b.initMatrix(&(p->b),sup_par.layer_n[k+1],1);
        p->Z.initMatrix(&(p->Z),sup_par.layer_n[k+1],X.row);


        for(i=0; i<p->W.col; i++)
        {
            for(j=0; j<p->W.row; j++)
            {
                radom=(rand()%100)/100.0;
                p->W.mat[i][j]=radom/sqrt(sup_par.layer_n[k]);//一种常用的参数初始化方法，参数初始化也有技巧
            }
        }
        //p->W.print(p->W);
        //cout<<"---------------------------------------"<<endl;
        p->next=new parameters();//下一层网络参数
        p->next->pre=p;
        p=p->next;

        g->grad_A.initMatrix(&(g->grad_A),sup_par.layer_n[L-k-1],X.row);
        g->grad_Z.initMatrix(&(g->grad_Z),sup_par.layer_n[L-k-1],X.row);
        g->grad_W.initMatrix(&(g->grad_W),sup_par.layer_n[L-k-1],sup_par.layer_n[L-k-2]);
        g->grad_b.initMatrix(&(g->grad_b),sup_par.layer_n[L-k-1],1);


        g->pre=new grad();//上一层网络参数梯度
        g->pre->next=g;
        g=g->pre;


        p->A.initMatrix(&(p->A),sup_par.layer_n[k+1],X.row);
        p->AT.initMatrix(&(p->AT),X.row,sup_par.layer_n[k+1]);
    }
    g->grad_A.initMatrix(&(g->grad_A),sup_par.layer_n[L-k-1],X.row);

    return 0;
}

void line_forward(parameters *p)
{
    int i=0,j=0;
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

void line_active_forward(parameters *p,string active)
{
    line_forward(p);
    if(active=="relu")
    {
        relu_forward(p);
    }
    if(active=="sigmoid")
    {
        sigmoid_forward(p);
    }
}

Matrix model_forward(Matrix X)
{
    int i=0;
    int L=sup_par.layer_dims;

    parameters *p=&par;
    for(i=0; i<L-1&&p->next!=NULL; i++)
    {
        line_active_forward(p,sup_par.layer_active[i+1]);
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

void line_backword(parameters *p,grad *g)
{
    int i,j;

    p->AT.transposematrix(p->A,&p->AT);

    g->grad_W.multsmatrix(&g->grad_W,g->grad_Z,p->AT);

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
}

void line_active_backword(parameters *p,grad *g,string active)
{
    if(active=="sigmoid")
    {
        sigmoid_backword(p,g);
        line_backword(p->pre,g);
    }
    if(active=="relu")
    {
        relu_backword(p,g);
        line_backword(p->pre,g);
    }
}
void model_backword(Matrix AL,Matrix Y)
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
        line_active_backword(p,g,sup_par.layer_active[i]);
        g=g->pre;
        p=p->pre;
    }
}

double cost_cumpter(Matrix AL,Matrix Y)
{
    int i=0;
    int n=Y.row;
    double loss=0;
    //AL.print(AL);
    for(i=0; i<n; i++)
    {
        loss+=-(Y.mat[0][i]*log(AL.mat[0][i])+(1-Y.mat[0][i])*log(1-AL.mat[0][i]));
    }
    return loss/n;
}

void updata_parameters(double learn_rateing)
{
    int k=0,i=0,j=0;
    int L=sup_par.layer_dims;
    parameters *p=&par;
    grad *g=&gra;
    while(g->pre->pre!=NULL)
    {
        g=g->pre;
    }
    for(k=0; k<L-1&&p!=NULL&&g!=NULL; k++)
    {
        for(i=0; i<g->grad_W.col; i++)
        {
            g->grad_b.mat[i][j]*=-learn_rateing;
            for(j=0; j<g->grad_W.row; j++)
            {
                g->grad_W.mat[i][j]*=-learn_rateing;
            }
        }
        p->W.addmatrix(&p->W,p->W,g->grad_W);
        p->b.addmatrix(&p->b,p->b,g->grad_b);
        cout<<"-------"<<endl;
        p->b.print(p->b);
        cout<<"-------"<<endl;
        cout<<"######"<<endl;
        g->grad_b.print(g->grad_b);
        cout<<"-#####"<<endl;
        p=p->next;
        g=g->next;
    }
}

int DNN(Matrix X,Matrix Y,double learn_rateing,int iter)
{
    int i=0;
    double loss;

    Matrix AL;
    AL.initMatrix(&AL,Y.col,Y.row);

    for(i=0; i<iter&&i<100; i++)
    {
        //cout<<"-----------forward------------"<<"i="<<i<<endl;
        AL=model_forward(X);

        //cout<<"-----------loss--------------"<<endl;
        loss=cost_cumpter(AL,Y);

        cout<<"loss="<<loss<<endl;

       // cout<<"-----------backword-----------"<<endl;
        model_backword(AL,Y);

       // cout<<"-----------update--------------"<<endl;
        updata_parameters(learn_rateing);
        cout<<"-----------update-----------"<<endl;
    }
    return 0;
}

int main()
{
    int i=0;
    int lay_dim=3;
    int lay_n[3]= {12288,5,1};
    string lay_active[3]= {"relu","relu","sigmoid"};
    double learn_rateing=0.001;
    int iter=1000;

    sup_par.layer_dims=lay_dim;
    for(i=0; i<lay_dim; i++)
    {
        sup_par.layer_n[i]=lay_n[i];
        sup_par.layer_active[i]=lay_active[i];
    }

    /**
        加载数据
    **/
    dataToMatrix dataset;
    dataset.loadData(&dataset,"datasets//test.txt");

    /**
        将数据转换成矩阵形式

    **/
    Matrix train_data;
    train_data.loadMatrix(&train_data,dataset);

    //Matrix train_dataT;
    //train_dataT.initMatrix(&train_dataT,train_data.row,train_data.col);
    //train_data.transposematrix(train_data,&train_dataT);

    /**
        生成输入输出矩阵
    **/
    Matrix X;
    Matrix Y;
    X.initMatrix(&X,train_data.col,train_data.row);
    Y.initMatrix(&Y,1,train_data.row);
    X.copy(train_data,&X);
    X.deleteOneCol(&X,X.col);
    Y=Y.getOneCol(train_data,train_data.col);

    /**
        构建网络结构
        初始化参数

    **/
    init_parameters(X);
    /**
        神经网络调用

    **/
    DNN(X,Y,learn_rateing=0.001,iter=5000);

    /****/

    return 0;
}

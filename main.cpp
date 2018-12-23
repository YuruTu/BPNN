//作者cclplus
//初稿2018/05/01
//如果你认为有必要打赏我，我的支付宝号是707101557@qq.com
#include "pch.h"
using namespace std;
const double pi = atan(1.0) * 4;
//BP神经网络结构
struct BPNN {
	int sample_count;//样本数量
	int input_count;//输入向量的维数
	int output_count;//输出向量的维数
	int hidden_count;//实际使用隐层神经元数量
	double study_rate;//学习速率
	double precision;//精度控制参数
	int loop_count;//循环次数
	vector<vector<double>> v;//隐含层权矩阵
	vector<vector<double>> w;//输出层权矩阵
};
BPNN CREATE_BPNN(int sc, int ic, int oc, int hc, double sr, double p, int lc);//创建一个BP神经网络
double rand_normal();//返回一个double类型的随机数
double sigmoid(double net) {
	return 1.0 / (1 + exp(-net));
}
double purelin(double net) {
	return 1.0 / (1 + exp(-net));
}
BPNN train_bp(vector<vector<double>> x, vector<vector<double>> y, BPNN bp);//训练
void use_bp(BPNN bp, vector<vector<double>> inoput);//使用BP神经网络进行前向传导运算
int main() {
	//样本数目
	int sample_count = 100;
	//输入向量维数
	int input_count = 1;
	//输出向量维数
	int output_count = 1;
	//实际使用隐层神经元数目
	int hidden_count = 4;
	//学习速率
	double study_rate = 0.02;
	//精度控制参数
	double precision = 0.001;
	//循环次数
	int loop_count = 10000;
	int i;
	double temp, temp1;
	//训练样本
	vector<vector<double>> x;
	vector<vector<double>> y;
	vector<vector<double>> input;
	x.resize(sample_count);
	for (i = 0; i < sample_count; i++) {
		x[i].resize(input_count);
	}
	y.resize(sample_count);
	for (i = 0; i < sample_count; i++) {
		y[i].resize(output_count);
	}
	input.resize(sample_count);
	for (i = 0; i < sample_count; i++) {
		input[i].resize(input_count);
	}
	//自定义输入与输出
	for (i = 0; i < sample_count; i++) {
		temp = (double)i;
		temp1 = (double)sample_count;
		x[i][0] = pi / temp1 * temp;
		input[i][0] = pi / temp1 * temp;
		y[i][0] = 1.0*sin(x[i][0]);
	}
	BPNN bp;
	bp = CREATE_BPNN(sample_count, input_count, output_count, hidden_count, study_rate, precision, loop_count);
	bp = train_bp(x, y, bp);

	use_bp(bp, input);
	return 0;
}
//使用BP神经网络进行前向传导运算
void use_bp(BPNN bp, vector<vector<double>> input) {
	//设置临时变量
	double temp;
	int  i, j, k;
	vector<double> O1;
	O1.resize(bp.hidden_count);
	vector<vector<double>> output;
	output.resize(100);
	for (i = 0; i < 100; i++) {
		output[i].resize(bp.output_count);
	}
	for (i = 0; i < 100; i++) {
		for (j = 0; j < bp.hidden_count; j++) {
			temp = 0;
			for (k = 0; k < bp.input_count; k++) {
				temp = temp + input[i][k] * bp.v[k][j];
			}
			O1[j] = sigmoid(temp);
		}
		for (j = 0; j < bp.output_count; j++) {
			temp = 0;
			for (k = 0; k < bp.hidden_count; k++) {
				temp = temp + O1[k] * bp.w[k][j];
			}
			output[i][j] = sigmoid(temp);
		}
	}

	for (i = 0; i < 100; i++) {
		for (j = 0; j < bp.output_count; j++) {
			printf("%f    ", output[i][j]);
		}
	}
	printf("\n结束\n");
}

//训练一个BP神经网络
BPNN train_bp(vector<vector<double>> x, vector<vector<double>> y, BPNN bp) {
	double f, a;
	int hc, sc, lc, ic, oc;

	f = bp.precision;//精度控制参数
	a = bp.study_rate;//学习速率
	hc = bp.hidden_count;//隐含层数
	sc = bp.sample_count;//训练样本总数
	lc = bp.loop_count;//循环次数
	ic = bp.input_count;//输入维度
	oc = bp.output_count;//输出维度
	//修改量矩阵
	vector<double> chg_h;//隐层
	chg_h.resize(hc);
	vector<double> chg_o;//输出层
	chg_o.resize(oc);
	vector<double> O1;
	O1.resize(hc);
	vector<double> O2;
	O2.resize(oc);

	//临时变量
	double temp;
	int i, j, m, n;
	double mse;//均方误差
	double e;//误差
	e = f + 1;//保证循环顺利执行
	for (n = 0; (e > f) && (n < lc); n++) {//n代表循环次数
		e = 0;
		mse = 0;
		//全部样本均加入神经网络的训练
		for (i = 0; i < sc; i++) {
			//计算隐层输出向量
			for (m = 0; m < hc; m++) {
				temp = 0;
				for (j = 0; j < ic; j++) {
					temp = temp + x[i][j] * bp.v[j][m];
				}
				O1[m] = sigmoid(temp);
			}
			//计算输出值
			for (m = 0; m < oc; m++) {
				temp = 0;
				for (j = 0; j < hc; j++) {
					temp = temp + O1[j] * bp.w[j][m];
				}
				O2[m] = purelin(temp);
			}
			//计算输出层的权重修改
			for (j = 0; j < oc; j++) {
				chg_o[j] = O2[j] * (1 - O2[j])*(y[i][j] - O2[j]);
			}
			//计算隐层的权重修改
			for (j = 0; j < hc; j++) {
				temp = 0;
				for (m = 0; m < oc; m++) {
					temp = temp + bp.w[j][m] * chg_o[m];
				}
				chg_h[j] = temp * O1[j] * (1 - O1[j]);
			}
			//计算误差和均方根误差
			for (j = 0; j < oc; j++) {
				e = e + (y[i][j] - O2[j])*(y[i][j] - O2[j]);
				mse = mse + y[i][j] * y[i][j];
			}
			//对权值矩阵做出修改
			for (j = 0; j < hc; j++) {
				for (m = 0; m < oc; m++) {
					bp.w[j][m] = bp.w[j][m] + a * O1[j] * chg_o[m];
				}
			}
			for (j = 0; j < ic; j++) {
				for (m = 0; m < hc; m++) {
					bp.v[j][m] = bp.v[j][m] + a * x[i][j] * chg_h[m];
				}
			}
		}
		//每循环一百次输出一次误差
		if (n % 100 == 0) {
			mse = e / mse;
			printf("误差        :%f\n", e);
			printf("均方根误差  :%f\n", mse);
			printf("当前循环次数:%d\n", n);

		}
	}
	//循环结束，输出最终信息
	printf("循环总次数:%d\n", n);
	printf("调整后的隐层权值矩阵:\n");
	for (i = 0; i < ic; i++) {
		for (j = 0; j < hc; j++) {
			printf("%f    ", bp.v[i][j]);
		}
		printf("\n");
	}
	printf("调整后的输出层权值矩阵:\n");
	for (i = 0; i < hc; i++) {
		for (j = 0; j < oc; j++) {
			printf("%f    ", bp.w[i][j]);
		}
		printf("\n");
	}
	printf("神经网络训练结束:\n");
	printf("最终误差:%f\n", e);
	return bp;
}

//创建一个BP神经网络
BPNN CREATE_BPNN(int sc, int ic, int oc, int hc, double sr, double p, int lc) {
	BPNN bp;
	bp.sample_count = sc;
	bp.input_count = ic;
	bp.output_count = oc;
	bp.hidden_count = hc;
	bp.study_rate = sr;
	bp.precision = p;
	bp.loop_count = lc;
	int i, j;
	bp.v.resize(ic);//隐含层的权值矩阵，共有input_count行，hidden_count列
	for (i = 0; i < ic; i++) {
		bp.v[i].resize(hc);
	}
	//数据的初始化
	for (i = 0; i < ic; i++) {
		for (j = 0; j < hc; j++) {
			bp.v[i][j] = rand_normal();
		}
	}
	bp.w.resize(hc);//输出层的权值矩阵，共有hidden_count行，output_count列
	for (i = 0; i < hc; i++) {
		bp.w[i].resize(oc);
	}
	for (i = 0; i < hc; i++) {
		for (j = 0; j < oc; j++) {
			bp.w[i][j] = rand_normal();
		}
	}
	return bp;

}

//返回一个double类型的随机数，这么做的目的是破坏神经网络结构的对称性
//基本原理，参见独立同分布的中心极限定理
double rand_normal() {
	int  i;
	const int normal_count = 200;//样本数目采用200个
	double ccl_num;
	double ccl_s;
	double ccl_ar[normal_count];
	ccl_num = 0;

	for (i = 0; i < normal_count; i++) {
		ccl_ar[i] = rand() % 1000 / (double)1001;
		ccl_num += ccl_ar[i];
	}
	ccl_num -= (normal_count / 2);//减去0-1均匀分布的均值
	ccl_s = 1.0*normal_count / 12.0;//0-1分布的方差为1/12
	ccl_s = sqrt(ccl_s);
	ccl_num /= ccl_s;//此时ccl_num接近标准正态分布的一个子集
	ccl_num /= 100;//变为正态分布（0，0.01）的一个子集，论文中有给出证明过程
	cout << " 随机值" << ccl_num << endl;
	return ccl_num;
}

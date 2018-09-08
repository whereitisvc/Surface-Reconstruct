#include <opencv2\core.hpp>
#include <opencv2\imgcodecs.hpp> // imread
#include <opencv2\highgui.hpp> // imshow, waitKey
#include <iostream>
#include <fstream>
using namespace std;
using namespace cv;

//image's total rows and columns
int rows;
int cols;

void outputPLY(String name, Mat z){
	fstream fw;
	fw.open(name, ios::out);
	if (!fw){
		cout << "Fail to open file" << endl;
	}
	fw << "ply" << endl
		<< "format ascii 1.0" << endl
		<< "comment alpha=1.0" << endl
		<< "element vertex " << rows*cols << endl
		<< "property float x" << endl
		<< "property float y" << endl
		<< "property float z" << endl
		<< "property uchar red" << endl
		<< "property uchar green" << endl
		<< "property uchar blue z" << endl
		<< "end_header" << endl;
	for (int ri = 0; ri < rows; ri++){
		for (int ci = 0; ci < cols; ci++){
			fw << ri << " " << ci << " " << z.at<float>(ri, ci) << " 255 255 255" << endl;
		}
	}
	fw.close();
}

int main() {

	int picNum = 6;

	//Read 6 pictures' light source vector (normalized), S
	FILE *fp = fopen("bunny/LightSource.txt", "r");
	char str1[10];
	float x, y, z;
	int index = 0;
	Mat S(Mat::zeros(picNum, 3, CV_32F));
	while (fscanf(fp, "%s (%f,%f,%f)", &str1, &x, &y, &z) != EOF){
		float len = sqrt(x*x + y*y + z*z);
		S.at<float>(index, 0) = x / len;
		S.at<float>(index, 1) = y / len;
		S.at<float>(index, 2) = z / len;
		index++;
	}

	//Read 6 pictures
	Mat img = imread("bunny/pic1.bmp", IMREAD_GRAYSCALE);
	Mat img2 = imread("bunny/pic2.bmp", IMREAD_GRAYSCALE);
	Mat img3 = imread("bunny/pic3.bmp", IMREAD_GRAYSCALE);
	Mat img4 = imread("bunny/pic4.bmp", IMREAD_GRAYSCALE);
	Mat img5 = imread("bunny/pic5.bmp", IMREAD_GRAYSCALE);
	Mat img6 = imread("bunny/pic6.bmp", IMREAD_GRAYSCALE);
	rows = img.rows; cols = img.cols;

	//Caculate each pixel's normal vector
	Mat N(Mat::zeros(img.rows, img.cols, CV_32FC3));         // each pixel's normal vector
	Mat N_albedos(Mat::zeros(img.rows, img.cols, CV_32FC3)); // each pixel's normal vector*albedos
	Mat albedos(Mat::zeros(img.rows, img.cols, CV_32F));     // each pixel's albedos
	Mat b(Mat::zeros(1, 3, CV_32F));						 // pixel's normal vector * albedos
	Mat I(Mat::zeros(picNum, 1, CV_32F));					 // pixel's intensity at (x,y) (6 pictures)
	for (int ri = 0; ri < img.rows; ri++){
		for (int ci = 0; ci < img.cols; ci++){

			I.at<float>(0, 0) = img.at<uchar>(ri, ci);
			I.at<float>(1, 0) = img2.at<uchar>(ri, ci);
			I.at<float>(2, 0) = img3.at<uchar>(ri, ci);
			I.at<float>(3, 0) = img4.at<uchar>(ri, ci);
			I.at<float>(4, 0) = img5.at<uchar>(ri, ci);
			I.at<float>(5, 0) = img6.at<uchar>(ri, ci);

			b = (S.t()*S).inv() * S.t() * I;

			float x1 = b.at<float>(0, 0);
			float x2 = b.at<float>(1, 0);
			float x3 = b.at<float>(2, 0);
			float length = sqrt(x1*x1 + x2*x2 + x3*x3);
			albedos.at<float>(ri, ci) = length;

			if (length > 0){
				N.at<Vec<float, 3>>(ri, ci) = Vec<float, 3>(x1 / length, x2 / length, x3 / length);
				N_albedos.at<Vec<float, 3>>(ri, ci) = Vec<float, 3>(x1, x2, x3);
			}
			else
				N.at<Vec<float, 3>>(ri, ci) = Vec<float, 3>(0, 0, 0);
		}
	}

	//Record each pixel's x,y gradient
	Mat G(Mat::zeros(img.rows, img.cols, CV_32FC3)); 
	float n1, n2, n3;     // represent pixel's normal vector (n1,n2,n3)
	for (int ri = 0; ri < img.rows; ri++){
		for (int ci = 0; ci < img.cols; ci++){
			n1 = N.at<Vec<float, 3>>(ri, ci)[0];
			n2 = N.at<Vec<float, 3>>(ri, ci)[1];
			n3 = N.at<Vec<float, 3>>(ri, ci)[2];
			if (n3 != 0){
				G.at<Vec<float, 2>>(ri, ci)[0] = (-n1 / n3); // x gradient
				G.at<Vec<float, 2>>(ri, ci)[1] = (-n2 / n3); // y gradient
			}
			else G.at<Vec<float, 2>>(ri, ci) = Vec<float, 2>(0, 0);
		}
	}


	// Surface reconstruction
													// record each pixel's depth
	Mat Z(Mat::zeros(img.rows, img.cols, CV_32F));  // final
	Mat Zm(Mat::zeros(img.rows, img.cols, CV_32F)); // integrate from mid
	Mat Z1(Mat::zeros(img.rows, img.cols, CV_32F)); // integrate from buttom, first y then x axis
	Mat Z2(Mat::zeros(img.rows, img.cols, CV_32F)); // integrate from head,   first y then x axis
	Mat Z3(Mat::zeros(img.rows, img.cols, CV_32F)); // integrate from head,   firat x then y axis
	Mat Z4(Mat::zeros(img.rows, img.cols, CV_32F)); // integrate from buttom, first x then y axis

	float last_r = img.rows - 1, last_c = img.cols - 1;
	float mid_r = img.rows / 2, mid_c = img.cols / 2;
	float gx, gy;
	float zx = 0, zy = 0, zsum;

	for (int ri = 0; ri < img.rows; ri++){
		for (int ci = 0; ci < img.cols; ci++){

			////////////////////////////////////////////////////// mid (Zm)
			// accumlate gradient x
			if (mid_c > ci){
				for (int index = mid_c; index > ci; index--){
					gx = G.at<Vec<float, 2>>(ri, index)[0];
					zx = zx + gx;
				}
			}
			else{
				for (int index = mid_c; index < ci; index++){
					gx = G.at<Vec<float, 2>>(ri, index)[0];
					zx = zx + gx;
				}
			}
			// accumlate gradient y
			if (mid_r > ri){
				for (int index = mid_r; index > ri; index--){
					gy = G.at<Vec<float, 2>>(index, mid_c)[1];
					zy = zy + gy;
				}
			}
			else{
				for (int index = mid_r; index < ri; index++){
					gy = G.at<Vec<float, 2>>(index, mid_c)[1];
					zy = zy + gy;
				}
			}
			zsum = zx + zy;
			if (zsum < 0) zsum = zsum*-1;
			Zm.at<float>(ri, ci) = zsum;
			zx = 0; zy = 0;

			////////////////////////////////////////////////////// left (Z1)
			// accumlate gradient x
			for (int index = last_c; index > ci; index--){
				gx = G.at<Vec<float, 2>>(ri, index)[0];
				zx = zx + gx;
			}

			// accumlate gradient y
			for (int index = last_r; index > ri; index--){
				gy = G.at<Vec<float, 2>>(index, last_c)[1];
				zy = zy + gy;
			}
			zsum = zx + zy;
			if (zsum < 0) zsum = zsum*-1;
			Z1.at<float>(ri, ci) = zsum;
			zx = 0; zy = 0;


			////////////////////////////////////////////////////// right (Z2)
			// accumlate gradient x
			for (int index = 0; index < ci; index++){
				gx = G.at<Vec<float, 2>>(ri, index)[0];
				zx = zx + gx;
			}

			// accumlate gradient y
			for (int index = 0; index < ri; index++){
				gy = G.at<Vec<float, 2>>(index, 0)[1];
				zy = zy + gy;
			}
			zsum = zx + zy;
			//if (zsum < 0) zsum = zsum*-1;
			Z2.at<float>(ri, ci) = zsum;
			zx = 0; zy = 0;

			/////////////////////////////////////////////////// up (Z3)
			// accumlate gradient x
			for (int index = 0; index < ci; index++){
				gx = G.at<Vec<float, 2>>(0, index)[0];
				zx = zx + gx;
			}

			// accumlate gradient y
			for (int index = 0; index < ri; index++){
				gy = G.at<Vec<float, 2>>(index, ci)[1];
				zy = zy + gy;
			}
			zsum = zx + zy;
			if (zsum < 0) zsum = zsum*-1;
			Z3.at<float>(ri, ci) = zsum;
			zx = 0; zy = 0;

			///////////////////////////////////////////////// down (Z4)
			// accumlate gradient x
			for (int index = last_c; index > ci; index--){
				gx = G.at<Vec<float, 2>>(last_r, index)[0];
				zx = zx + gx;
			}

			// accumlate gradient y
			for (int index = last_r; index > ri; index--){
				gy = G.at<Vec<float, 2>>(index, ci)[1];
				zy = zy + gy;
			}
			zsum = zx + zy;
			//if (zsum < 0) zsum = zsum*-1;
			Z4.at<float>(ri, ci) = zsum;
			zx = 0; zy = 0;

		}
	}

	//Empty background, depth = 0
	for (int ri = 0; ri < img.rows; ri++){
		for (int ci = 1; ci < img.cols; ci++){
			if (albedos.at<float>(ri, ci) == 0){
				Zm.at<float>(ri, ci) = 0;
				Z1.at<float>(ri, ci) = 0;
				Z2.at<float>(ri, ci) = 0;
				Z3.at<float>(ri, ci) = 0;
				Z4.at<float>(ri, ci) = 0;
			}
		}
	}

	//Combine the result of different integrate path (for better result)
	double alpha = 1, beta = 1;
	double ratio = 0.99;
	for (int ri = 0; ri < img.rows; ri++){
		for (int ci = 0; ci < img.cols; ci++){
			float zm = Zm.at<float>(ri, ci);
			float z1 = Z1.at<float>(ri, ci);
			float z2 = Z2.at<float>(ri, ci);
			float z3 = Z3.at<float>(ri, ci);
			float z4 = Z4.at<float>(ri, ci);
			Z.at<float>(ri, ci) = (z2*beta + z1*(1 - beta)) * 0.5 + (z3*alpha + z4*(1 - alpha))*0.5; // best for bunny.bmp
			//Z.at<float>(ri, ci) = (z2*beta + z1*(1 - beta)); //best for venus.bmp
			//Z.at<float>(ri, ci) = (z3*alpha + z4*(1 - alpha));
			beta = beta * ratio;
		}
		alpha = alpha * ratio;
		beta = 1;
	}

	//Output the result as .ply
	outputPLY("bunny.ply", Z);
	outputPLY("bunny_left.ply", Z1);
	outputPLY("bunny_right.ply", Z2);
	outputPLY("bunny_up.ply", Z3);
	outputPLY("bunny_down.ply", Z4);
	outputPLY("bunny_mid.ply", Zm);

	//Write normal map image
	imwrite("bunny_normalmap.bmp", N_albedos);


	system("pause");
	return 0;

}
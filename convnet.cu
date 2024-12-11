#include "library.cuh"
#include <cstdio>
#include <vector>
#include <iostream>
// #include <torch/script.h>


using namespace std;
// using namespace torch::jit::script;
cudaEvent_t st, stop;

void startTiming() {
    cudaEventCreate(&st);
    cudaEventCreate(&stop);
    cudaEventRecord(st);
}

double stopTiming() {
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, st, stop);
    cudaEventDestroy(st);
    cudaEventDestroy(stop);
    return (double)milliseconds ;
}
const int N = 1 << 15;
const double scale = 1 << 30;
const int moduleSize = 6;
const int ci = 32;
const int co = 32;
const int f = 1;
const int f2 = f * f;
const int dataSize = N * 2 * moduleSize * sizeof(unsigned long long);
const int H = 16;
const int W = 16;

const int g = 2;
const int d = 4;
const int downSize = 2;
void getkerCA(vector<unsigned long long*> &kerVecCA, double ker[co][ci][f][f], int ci, int co, int f, int H, int W){
    
    double tmp[N];
    for(int i = 0; i < N; i++)tmp[i] = 0;
    for(int i = 0; i < max(co / d,1); i++){
        for(int k = 0; k < f; k++){
            for(int l = 0; l < f; l++){
                for(int j = 0; j < ci; j++){
                    for(int m = 0; m < H; m++){
                        for(int n = 0; n < W; n++){
                            for(int o = 0; o < d; o++){
                                // int x = j % g + j / g * W * g + n * g;
                                // int y = m * d + o;
                                // tmp[y * ci * W + x] = ker[i * d + o%co][j][k][l];
                                // tmp[y * ci * W + x] = l + k * f + j * f2 + (i * d + o) * f2 * ci;

                                // tmp[j * W + m * W * ci + n ] = ker[i][j][k][l];

                                int x = j % g + n * g;
                                int y = m * d + o + j / g * d * H;
                                tmp[y * g * W + x] = ker[i * d + o%co][j][k][l];
                                if(n + l >= W || m + k >= H){
                                    tmp[y * g * W + x] = 0;
                                }

                            }
                        }
                    }
                }


                // if(kerVecCA.size() == 0){
                //     for(int i = 0; i < d * H; i++){
                //     for(int j = 0; j < W * g; j++){
                //         printf("%4.0lf ",tmp[i * W * g + j]);
                //     }
                //     printf("\n");
                // }       
                //      }
                    //  exit(1);
                auto res = encode(tmp);
                kerVecCA.push_back(res);
            }
        }
    }
    if(downSize == 2){
        int size = kerVecCA.size();
        for(int i = 0; i < size; i++){
            kerVecCA.push_back(kerVecCA[i]);
        }
    }
}
void getMaskCA(double* mask, int ci, int H, int W){
    for(int i = 0; i < N; i++)mask[i] = 0;
    for(int i = 0; i < N; i++){

        int j = i / ( H * g * W * d);
        if(  i % (g * downSize)== 0){
            mask[i] = 1;
        }
    }
}

void getMaskRA(double* mask, int ci, int H, int W){
    for(int i = 0; i < N; i++)mask[i] = 0;
    for(int i = 0; i < N; i++){

        int j = i / (g * W);
        if(j % d== 0){
            mask[i] = 1;
        }
    }
}


void getker(double ker[co][ci][f][f]){

    // Module model;
    // model = torch::jit::load("resnet20.pt");

    // Module conv1 = model.attr("conv2").toModule();

    // auto params = *conv1.parameters().begin();

    
    for(int l = 0; l < co; l++){
        for(int i = 0; i < ci; i++){
            for(int j = 0; j < f; j++){
                for(int k = 0; k < f; k++){
                    // ker[l][i][j][k] = params[l][i][j][k].item<double>();
                    ker[l][i][j][k] = rand()%2;
                    ker[l][i][j][k] = k + j * f + i * f2 + l * f2 * ci;
                }
            }
        }
    }
}

void sumUp(vector<cipherText> &cipherVec, int c, int f2){
    for(int i = 0; i < c; i++){
        for(int j = 0; j < f2; j++){
            addcipher(cipherVec[i * f2 + 0],cipherVec[i * f2 + j]);
        }
    }
}



void rotFCA(vector<cipherText> &cipherVec, int s, int f){
    for(int i = 1; i < f; i++){
        rotation_to_another(cipherVec[(i-1) * f],cipherVec[f * i], s,getgalois());
    }
    for(int i = 0; i < f; i++){
        for(int j = 1; j < f; j++){
            rotation_to_another(cipherVec[i * f + j - 1],cipherVec[i * f + j], g,getgalois());
        }
    }
}
void rotFRA(vector<cipherText> &cipherVec, int s, int f){

    for(int i = 0; i < f; i++){
        for(int j = 0; j < f; j++){
            int allstep = i * s + j * g;

            int step = 1;
            while(allstep){
                if(allstep & 1){
                    rotation_to_another(cipherVec[i * f + j],cipherVec[i * f + j], step ,getgalois());
                }
                step *= 2;
                allstep /= 2;
            }
        }
    }
}

void dupCA(vector<cipherText> &cipherVec, int c, int f2){
    for(int i = 1; i < c; i++){
        for(int j = 0; j < f2; j++){
            copyCipher(cipherVec[i*f2+j],cipherVec[j],dataSize);
        }
    }
}
void dupRA(vector<cipherText> &cipherVec, int c, int f2){
    for(int i = 0; i < c; i++){
        for(int j = 1; j < f2; j++){
            copyCipher(cipherVec[i*f2+j],cipherVec[i*f2],dataSize);
        }
    }
}
void sumCA(vector<cipherText> &cipherVec, int c, int f2){
    if(c==0)c=1;
    for(int i = 0; i < c; i++){
        for(int j = 1; j < f2; j++){
            addcipher(cipherVec[i*f2 + 0],cipherVec[i*f2+j]);
        }
    }
}
void rasCA(vector<cipherText> &cipherVec, cipherText &tmp, int co, int ci, int f2){
    for(int i = 0; i < max(downSize * co/d,1); i++){
        for(int s = 1; s < ci/g; s*=2){
            rotation_to_another(cipherVec[i * f2],tmp, s*W*g * H * d,getgalois());
            addcipher(cipherVec[i * f2],tmp);
        }        
    }
    for(int i = 0; i < max(downSize * co/d,1); i++){
        for(int s = 1; s < g; s*=2){
            rotation_to_another(cipherVec[i * f2],tmp, s,getgalois());
            addcipher(cipherVec[i * f2],tmp);
        }        
    }
}
void rasRA(vector<cipherText> &cipherVec, cipherText &tmp, int co, int ci, int f2){
        
    for(int i = 0; i < f2; i++){    
        for(int s = 1; s < d; s*=2){
            rotation_to_another(cipherVec[i],tmp, s * W * g,getgalois());
            addcipher(cipherVec[i],tmp);
        }  
    }    
}
void irCA(vector<cipherText> &cipherVec, cipherText &tmp, int co, int ci, int f2){
    // for(int i = 0; i < co / d; i++){
    //     rotation_to_anotherinv(cipherVec[(i + co / d) * f2 ],tmp, d * H * g * W,getgaloisinv());
    //     addcipher(cipherVec[i * f2],tmp);  
    // }
    for(int i = 0; i < max(1,downSize * co/d); i++){
        for(int s = 1; s < g * downSize; s*=2){
            rotation_to_anotherinv(cipherVec[i * f2],tmp, s,getgaloisinv());
            addcipher(cipherVec[i * f2],tmp);
        }        
    }

}

void irRA(vector<cipherText> &cipherVec, cipherText &tmp, int co, int ci, int f2){
    for(int i = 1; i < d; i*=2){
        rotation_to_anotherinv(cipherVec[0],tmp, i * g * W,getgaloisinv());
        addcipher(cipherVec[0],tmp);  
    }

}
void maskCA(vector<cipherText> &cipherVec,  unsigned long long* encmask, int co, int ci, int f2){
    for(int i = 0; i < co; i++){
        mulPlain(cipherVec[i * f2], encmask);
    }
}
void maskRA(vector<cipherText> &cipherVec,  unsigned long long* encmask, int co, int ci, int f2){
    for(int i = 0; i < 1; i++){
        mulPlain(cipherVec[0], encmask);
    }
}
void mulall(vector<cipherText> &cipherVec, vector<unsigned long long*> kerVecCA){
    for(int i = 0; i < kerVecCA.size(); i++){
        mulPlain(cipherVec[i],kerVecCA[i]);
    }
}
void convCA(vector<cipherText> &cipherVec, cipherText tmp, vector<unsigned long long*> &kerVecCA,unsigned long long* encmask, int co, int ci, int f){
    int f2 = f * f;
    rotFCA(cipherVec, W * g * d, f);
    dupCA(cipherVec, downSize * co /d, f2);

    mulall(cipherVec, kerVecCA);
    sumCA(cipherVec, downSize * co/d, f2);
    rasCA(cipherVec, tmp, co, ci, f2);
    
    maskCA(cipherVec,encmask,downSize * co/d,ci,f2);
    irCA(cipherVec, tmp, co, ci, f2);

}


void getkerRA(vector<unsigned long long*> &kerVecRA, double ker[co][ci][f][f], int ci, int co, int f, int H, int W){
    
    double tmp[N];
    for(int i = 0; i < N; i++)tmp[i] = 0;

    for(int i = 0; i < max(co/d,1); i++){
        for(int k = 0; k < f; k++){
            for(int l = 0; l < f; l++){
                for(int j = 0; j < co; j++){
                    for(int m = 0; m < H; m++){
                        for(int n = 0; n < W; n++){
                            for(int o = 0; o < d; o++){
                                // int x = j % d + n * g;
                                // int y = m * d + o + j / g * d * H;
                                // int y = j + i * d;
                                int x = n * g + j % g;
                                int y = o + m * d + j / g * H * d;
                                tmp[y * g * W + x] = ker[j][i * d + o][k][l]; 
                                // tmp[(j * W + m * W * ci + n  )%N] = ker[i][j][k][l];
                                if( l > n || k > m){
                                    tmp[y * g * W + x] = 0;
                                }
                            }
                        }
                    }
                }
                // if(kerVecRA.size() == 0){
                //     printf("========\n\n\n\n");
                //     for(int i = 0; i < d * H * 2; i++){
                //     for(int j = 0; j < W * g; j++){
                //         printf("%4.0lf ",tmp[i * W * g + j]);
                //     }
                //     printf("\n");
                // }       
                //      }
                auto res = encode(tmp);
                for(int ii = 0; ii < N; ii++)tmp[ii] = 0;

                kerVecRA.push_back(res);
            }
        }
    }

}
void sumRA1(vector<cipherText> &cipherVec, int c, int f2){
    for(int i = 1; i < c; i++){
        for(int j = 0; j < f2; j++){
            addcipher(cipherVec[j],cipherVec[i*f2+j]);
        }
    }
}

void sumRA2(vector<cipherText> &cipherVec, int c, int f2){
    for(int i = 1; i < f2; i++){
        addcipher(cipherVec[0],cipherVec[i]);
    }

}
void convRA(vector<cipherText> &cipherVec, cipherText tmp, vector<unsigned long long*> &kerVecRA, unsigned long long* encmask, int co, int ci, int f){
    int f2 = f * f;
    dupRA(cipherVec, co/d, f2);
    mulall(cipherVec, kerVecRA);
    sumRA1(cipherVec, co/d, f2);

    rasRA(cipherVec, tmp, co, ci, f2);
    rotFRA(cipherVec, W * g * d , f);
    sumRA2(cipherVec, ci, f2);
    maskRA(cipherVec,encmask,downSize * co/d,ci,f2);
    irRA(cipherVec, tmp, co, ci, f2);
}

void convfirst(vector<cipherText> &cipherVec, cipherText tmp, vector<unsigned long long*> &kerVecRA, unsigned long long* encmask, int co, int ci, int f){
    int f2 = f * f;
    dupRA(cipherVec, co/d, f2);
    mulall(cipherVec, kerVecRA);
    sumRA1(cipherVec, co/d, f2);

    rasRA(cipherVec, tmp, co, ci, f2);
    rotFRA(cipherVec, W * g * d , f);
    sumRA2(cipherVec, ci, f2);
    maskRA(cipherVec,encmask,downSize * co/d,ci,f2);
    irRA(cipherVec, tmp, co, ci, f2);
}
void getimage(double* image, int ci, int H, int W){
    for(int i = 0; i < ci; i++){
        for(int j = 0; j < H; j++){
            for(int k = 0; k < W; k++){
                // image[i * W + j * W * ci + k] = rand() % 2;
            }
        }
    }



    for(int i = 0; i < ci; i++){
        for(int j = 0; j < H; j++){
            for(int k = 0; k < W; k++){
                for(int l = 0; l < d; l++){
                    int x = i % g + k * g;
                    int y = j * d + l + i / g * d * H;
                    image[x + y * W * g ] = k + j * W + i * H * W;
                    // cout << x + y * W * ci << endl;
                }
            }
        }
    }
    // for(int i = 0; i < d * H * ci / g; i++){
    //     for(int j = 0; j < g * W; j++){
    //         printf("%4.0lf ",image[i * W * g + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
    // exit(1);

}
void getimageRA(double* image, int ci, int H, int W){


    for(int i = 0; i < d; i++){
        for(int j = 0; j < H; j++){
            for(int k = 0; k < W; k++){
                for(int l = 0; l < g; l++){

                    for(int m = 0; m < co / g; m++){                    
                        int x = k * g + l;
                        int y = j * d + i + m * H * d; 
                        // if(x == 0 && y==2){
                        //     printf("%4d,%4d,%d,%d\n",i,j,k,l);
                        // }
                        image[y * g * W + x] = k + j * W + 0 * H * W;
                    }
                }
            }
        }
    }
    // printf("imageRA\n\n\n");
    // for(int i = 0; i < d * H * ci / g; i++){
    //     for(int j = 0; j < g * W; j++){
    //         printf("%4.0lf ",image[i * W * g + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
    // exit(1);

}
void testconv(double* image,  double ker[co][ci][f][f]){
    // double out[co][H][W];
    // std::cout << "image data" << endl;
    // for (int i = 0; i < ci; ++i) {
    //    for (int j = 0; j < H; ++j) {
    //        for (int k = 0; k < W; ++k) {
    //             int x = i % g + i / g * W * g + k * g;
    //             int y = j * d;
    //            std::cout << image[x + y * W * ci ] << " ";
    //        }
    //        std::cout << std::endl;
    //     }
    //    std::cout << std::endl;
    // }



    // std::cout << "ker data" << endl;

    // for(int i = 0; i < co; i++){
    //     for(int j = 0; j < ci; j++){
    //         for(int k = 0; k < f; k++){
    //             for(int l = 0; l < f; l++){
    //                 std::cout << ker[i][j][k][l] << " ";
    //             }
    //             std::cout << std::endl;
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }

    int padded_H = H + f - 1, padded_W = W + f - 1;
    double padded_image[ci][padded_H][padded_W] = {0};
    for(int i = 0; i < ci; i++){
        for(int j = 0; j < padded_H; j++){
            for(int k = 0; k < padded_W; k++){
                padded_image[i][j][k] = 0;
            }
        }
    }
    double out[co][H][W] = {0};
    for(int i = 0; i < co; i++){
        for(int j = 0; j < H; j++){
            for(int k = 0; k < W; k++){
                out[i][j][k] = 0;
            }
        }
    }
    for (int c = 0; c < ci; ++c)
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j){
                    int x = c % g + j * g;
                    int y = i * d + c / g * d * H;
                    // image[x + y * W * g ] = k + j * W + i * H * W;
                padded_image[c][i][j] = image[x + y * W * g ];

                padded_image[c][i][j] = 0 * H * W + i * W + j;
            }
        

    for (int oc = 0; oc < co; ++oc)
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j)
                for (int ic = 0; ic < ci; ++ic)
                    for (int ki = 0; ki < f; ++ki)
                        for (int kj = 0; kj < f; ++kj)
                            out[oc][i][j] += padded_image[ic][i + ki][j + kj] * ker[oc][ic][ki][kj];

    // for (int k = 0; k < ci; ++k) {
    //    for (int i = 0; i < H; ++i) {
    //        for (int j = 0; j < W; ++j) {
    //            std::cout << out[k][i][j] << " ";

    //        }
    //        std::cout << std::endl;
    //     }
    //    std::cout << std::endl;
    // }
    // for (int c = 0; c < ci; ++c){
    //     for (int i = 0; i < H; ++i){

    //         for (int j = 0; j < W; ++j){

    //             padded_image[c][i][j] = out[c][i][j];
    //             out[c][i][j] = 0;
    //         }
    //     }
    // }


    // for (int oc = 0; oc < co; ++oc)
    // for (int i = 0; i < H; ++i)
    //     for (int j = 0; j < W; ++j)
    //         for (int ic = 0; ic < ci; ++ic)
    //             for (int ki = 0; ki < f; ++ki)
    //                 for (int kj = 0; kj < f; ++kj){
                        
    //                     out[oc][i][j] += padded_image[ic][i + ki][j + kj] * ker[ic][oc][ki][kj];
    //                     if(oc == 0 && i == 0 && j == 1){
    //                         printf("%lf,%lf\n",padded_image[ic][i + ki][j + kj], ker[oc][ic][ki][kj]);
    //                     }
    //                 }
    // for (int k = 0; k < co; ++k) {
    //    for (int i = 0; i < H; ++i) {
    //        for (int j = 0; j < W; ++j) {
    //            std::cout << out[k][i][j] << " ";
    //        }
    //        std::cout << std::endl;
    //     }
    //    std::cout << std::endl;
    // }
}


int main(){

    srand(114514);
    vector<cipherText> cipherVec;
    vector<unsigned long long*> kerVecCA;
    vector<unsigned long long*> kerVecRA;




    init(N, scale, moduleSize);
    
    double a[N]={0.0};
    double maskCA[N]={0.0};
    double maskRA[N]={0.0};
    double ker[co][ci][f][f]={0.0};
    getimage(a,ci,H,W);
    getker(ker);
    getkerCA(kerVecCA,ker,ci,co,f,H,W);
    getkerRA(kerVecRA,ker,ci,co,f,H,W);
    getMaskCA(maskCA,ci,H,W);
    getMaskRA(maskRA,ci,H,W);

    getimageRA(a,ci,H,W);
    testconv(a,ker);
    auto encA = encode(a);
    auto encmask = encode(maskCA);
    auto encmaskRA = encode(maskRA);

    auto cipherA = encrypt(encA,getpub());

    for(int i = 0; i < co*f2; i++){
        cipherVec.push_back(encrypt(encA,getpub()));
    }
    startTiming();
    convCA(cipherVec, cipherA, kerVecCA, encmask, co, ci, f);
    // convRA(cipherVec, cipherA, kerVecRA, encmaskRA, co, ci, f);

    // convfirst(cipherVec, cipherA, kerVecRA, encmaskRA, co, ci, f);

    printf("Time: %lf\n",stopTiming());

    auto dec = decrypt(cipherVec[0], getpri());
    // auto plaina = decode(dec,4);

    double scale2 = scale * scale / 1073872897;
    double scale3 = scale2 * scale / 1073971201;
    double scale4 = scale3 * scale / 1074266113;
// if(0){
//     for(int i = 0; i < d * H * 4; i++){
//         for(int j = 0; j < g * W; j++){
//             if(1)printf("%6.0lf ",plaina[i * g * W + j].x / scale);
//         }
//         printf("\n");
//     }
//     printf("\n");
// }
    // for(int i = 0; i < 32 ; i++){
    //     printf("%d,%lf+%lf i\n",i,plaina[i].x/scale4+0.001,plaina[i].y/scale4);
    // }
}
#include "library.cuh"
#include <cstdio>
#include <vector>
#include <iostream>
// #include <torch/script.h>


using namespace std;
// using namespace torch::jit::script;

const int N = 1 << 12;
const double scale = 1 << 30;
const int moduleSize = 8;
const int ci = 4;
const int co = 4;
const int f = 2;
const int f2 = f * f;
const int dataSize = N * 2 * moduleSize * sizeof(unsigned long long);
const int H = 4;
const int W = 4;

const int g = 2;
const int d = 64;
void getkerCA(vector<unsigned long long*> &kerVecCA, double ker[co][ci][f][f], int ci, int co, int f, int H, int W){
    
    double tmp[N];
    for(int i = 0; i < N; i++)tmp[i] = 0;
    for(int i = 0; i < max(co / d,1); i++){
        for(int k = 0; k < f; k++){
            for(int l = 0; l < f; l++){
                for(int j = 0; j < ci; j++){
                    for(int m = 0; m < H; m++){
                        for(int n = 0; n < W; n++){
                            for(int o = 0; o < min(co,d); o++){
                                // int x = j % g + j / g * H * W * g + m * g * W + n * g;
                                // int y = o;
                                // tmp[o * H * W * ci + x] = ker[i * d + o][j][k][l];
                                
                                int x = (j % g * g + j / g) * d * H * W + m * H * d + n * d + o;
                                tmp[x] = ker[i * d + o][j][k][l];
                                // int x = j * d * H * W + o + m * H * d + n * d + o;

                                // int x = j % g + j / g * W * g + n * g;
                                // int y = m * d + o;
                                // tmp[y * ci * W + x] = ker[i * d + o][j][k][l];
                                // tmp[y * ci * W + x] = l + k * f + j * f2 + (i * d + o) * f2 * ci;

                                // tmp[j * W + m * W * ci + n ] = ker[i][j][k][l];
                                if(n + l >= W || m + k >= H){
                                    tmp[x] = 0;
                                }

                            }
                        }
                    }
                }


                if(kerVecCA.size() == 0){
                    printf("AAAAAAAA\n");
                    for(int i = 0; i < ci; i++){
                    for(int j = 0; j < d * W * H; j++){
                        printf("%2.0lf ",tmp[i * d * W * H + j]);
                    }
                    printf("\n");
                }       
                     }
                    //  exit(1);
                auto res = encode(tmp);
                kerVecCA.push_back(res);
            }
        }
    }
}
void getMaskCA(double* mask, int ci, int H, int W){
    for(int i = 0; i < N; i++)mask[i] = 0;
    for(int i = 0; i < ci; i++){
        for(int j = 0; j < H; j++){
            for(int k = 0; k < W; k++){
                if(i == 0) mask[i * W + j * W * ci + k] = 1;
                else mask[i * W + j * W * ci + k] = 0;
            }
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
        rotation_to_another(cipherVec[(i-1) * f],cipherVec[f * i], s * d,getgalois());
    }
    for(int i = 0; i < f; i++){
        for(int j = 1; j < f; j++){
            rotation_to_another(cipherVec[i * f + j - 1],cipherVec[i * f + j], d,getgalois());
        }
    }
}
void rotFRA(vector<cipherText> &cipherVec, int s, int f){

    for(int i = 0; i < f; i++){
        for(int j = 0; j < f; j++){
            int allstep = i * s + j;

            int step = 1;
            while(allstep){
                if(allstep & 1){
                    rotation_to_another(cipherVec[i * f + j],cipherVec[i * f + j], step,getgalois());
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
    for(int i = 0; i < co/d; i++){
        for(int s = 1; s < ci; s*=2){
            rotation_to_another(cipherVec[i * f2],tmp, s*d*H*W,getgalois());
            addcipher(cipherVec[i * f2],tmp);
        }        
    }
    // for(int i = 0; i < co/d; i++){
    //     for(int s = 1; s < g; s*=2){
    //         rotation_to_another(cipherVec[i * f2],tmp, s,getgalois());
    //         addcipher(cipherVec[i * f2],tmp);
    //     }        
    // }
}

void irCA(vector<cipherText> &cipherVec, cipherText &tmp, int co, int ci, int f2){
    for(int i = 0; i < co; i++){
        for(int s = 1; s < ci; s*=2){
            rotation_to_anotherinv(cipherVec[i * f2],tmp, s*W,getgaloisinv());
            addcipher(cipherVec[i * f2],tmp);
        }        
    }
}
void maskCA(vector<cipherText> &cipherVec,  unsigned long long* encmask, int co, int ci, int f2){
    for(int i = 0; i < co; i++){
        mulPlain(cipherVec[i * f2], encmask);
    }
}

void mulall(vector<cipherText> &cipherVec, vector<unsigned long long*> kerVecCA){
    for(int i = 0; i < kerVecCA.size(); i++){
        mulPlain(cipherVec[i],kerVecCA[i]);
    }
}
void convCA(vector<cipherText> &cipherVec, cipherText tmp, vector<unsigned long long*> &kerVecCA,unsigned long long* encmask, int co, int ci, int f){
    int f2 = f * f;
    rotFCA(cipherVec, W , f);
    dupCA(cipherVec, co/g, f2);

    mulall(cipherVec, kerVecCA);
    sumCA(cipherVec, co/d, f2);
    rasCA(cipherVec, tmp, co, ci, f2);
    
    // maskCA(cipherVec,encmask,co,ci,f2);
    // irCA(cipherVec, tmp, co, ci, f2);

}


void getkerRA(vector<unsigned long long*> &kerVecRA, double ker[co][ci][f][f], int ci, int co, int f, int H, int W){
    
    double tmp[N];
    for(int i = 0; i < N; i++)tmp[i] = 0;

    for(int i = 0; i < co; i++){
        for(int k = 0; k < f; k++){
            for(int l = 0; l < f; l++){
                for(int j = 0; j < ci; j++){
                    for(int m = 0; m < H; m++){
                        for(int n = 0; n < W; n++){
                            tmp[(j * W + m * W * ci + n  )%N] = ker[i][j][k][l];
                            if( l > n || k > m){
                                tmp[(j * W + m * W * ci + n  )%N] = 0;
                            }
                        }
                    }
                }
                // if(kerVecRA.size() == 1){
                //     printf("%d,%d,%d,%d\n",i,114514,k,l);
                //     for(int ll = 0; ll < 32; ll++){
                //         printf("%lf ", tmp[ll]);
                //     }
                // }
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
    dupRA(cipherVec, co, f2);
    mulall(cipherVec, kerVecRA);
    sumRA1(cipherVec, ci, f2);
    rotFRA(cipherVec, W * ci, f);
    sumRA2(cipherVec, ci, f2);
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
                    // int x = i % g + i / g * H * W * g + k * g + j * g * W;
                    // int y = l;
                    int x = (i % g * g + i / g) * d * H * W + j * H * d + k * d + l;
                    // int y = i % g
                    image[x] = k + j * W + i * H * W;
                    // cout << x + y * W * ci << endl;
                }
            }
        }
    }
    for(int i = 0; i < ci ; i++){
        for(int j = 0; j < d * H * W; j++){
            printf("%2.0lf ",image[i * d * W * H  + j]);
        }
        printf("\n");
    }
    printf("\n");
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
                int x = (c % g * g + c / g) * d * H * W + i * H * d + j * d;

                padded_image[c][i][j] = image[x];
            }
        

    for (int oc = 0; oc < co; ++oc)
        for (int i = 0; i < H; ++i)
            for (int j = 0; j < W; ++j)
                for (int ic = 0; ic < ci; ++ic)
                    for (int ki = 0; ki < f; ++ki)
                        for (int kj = 0; kj < f; ++kj)
                            out[oc][i][j] += padded_image[ic][i + ki][j + kj] * ker[oc][ic][ki][kj];

    for (int k = 0; k < ci; ++k) {
       for (int i = 0; i < H; ++i) {
           for (int j = 0; j < W; ++j) {
               std::cout << out[k][i][j] << " ";

           }
           std::cout << std::endl;
        }
       std::cout << std::endl;
    }
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
    // for (int k = 0; k < ci; ++k) {
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
    double ker[co][ci][f][f]={0.0};
    getimage(a,ci,H,W);
    getker(ker);
    getkerCA(kerVecCA,ker,ci,co,f,H,W);
    printf("%d\n",__LINE__);
    // getkerRA(kerVecRA,ker,ci,co,f,H,W);
    getMaskCA(maskCA,ci,H,W);
    testconv(a,ker);

    auto encA = encode(a);
    auto encmask = encode(maskCA);
    auto cipherA = encrypt(encA,getpub());

    for(int i = 0; i < co*f2; i++){
        cipherVec.push_back(encrypt(encA,getpub()));
    }

    convCA(cipherVec, cipherA, kerVecCA, encmask, co, ci, f);

    // convRA(cipherVec, cipherA, kerVecRA, encmask, co, ci, f);



    auto dec = decrypt(cipherVec[0], getpri());
    auto plaina = decode(dec,4);

    double scale2 = scale * scale / 1073872897;
    double scale3 = scale2 * scale / 1073971201;
    double scale4 = scale3 * scale / 1074266113;

    for(int i = 0; i < d; i++){
        for(int j = 0; j < ci * W * H; j++){
            printf("%2.0lf ",plaina[i * ci * W * H + j].x / scale);
        }
        printf("\n");
    }
    printf("\n");
    // for(int i = 0; i < 32 ; i++){
    //     printf("%d,%lf+%lf i\n",i,plaina[i].x/scale4+0.001,plaina[i].y/scale4);
    // }
}
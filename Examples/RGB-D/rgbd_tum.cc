/**
* This file is a modified version of ORB-SLAM2.<https://github.com/raulmur/ORB_SLAM2>
*
* This file is part of DynaSLAM.
* Copyright (C) 2018 Berta Bescos <bbescos at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/bertabescos/DynaSLAM>.
*
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include <unistd.h>
#include "pointcloudmapping.h"
#include<opencv2/core/core.hpp>

#include "Geometry.h"
#include "MaskNet.h"
#include <System.h>

using namespace std;

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    if(argc != 5 && argc != 6 && argc != 7)
    {
        cerr << endl << "Usage: ./rgbd_tum path_to_vocabulary path_to_settings path_to_sequence path_to_association (path_to_masks) (path_to_output)" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;
    string strAssociationFilename = string(argv[4]);          //./Examples/rgbd_dataset_freiburg2_desk_with_person/associations.txt
    LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);  //把路径和时间节点都放到三个向量中了

    // Check consistency in the number of images and depthmaps
    int nImages = vstrImageFilenamesRGB.size();
    std::cout << "nImages: " << nImages << std::endl;   

    if(vstrImageFilenamesRGB.empty())
    {
        cerr << endl << "No images found in provided path." << endl;
        return 1;
    }
    else if(vstrImageFilenamesD.size()!=vstrImageFilenamesRGB.size())
    {
        cerr << endl << "Different number of images for rgb and depth." << endl;
        return 1;
    }
    
    
    

    // Initialize CNN
    DynaSLAM::SegmentDynObject *MaskNet;   
    if (argc==6 || argc==7)
    {
        cout << "Loading Yolact. This could take a while..." << endl;
        MaskNet = new DynaSLAM::SegmentDynObject();   //new一个对象  用于目标分割
        cout << "Yolact loaded!深度学习部分已经完成" << endl;
    }
    
    
    
    

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::RGBD,true);    //初始化SLAM系统，准备进行各种线程

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);   //记录实验计算一帧的时间

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Dilation settings
    int dilation_size = 15;
    cv::Mat kernel = getStructuringElement(cv::MORPH_ELLIPSE,
                                           cv::Size( 2*dilation_size + 1, 2*dilation_size+1 ),
                                           cv::Point( dilation_size, dilation_size ) );

    //用于膨胀与腐蚀的kernel
    
    if (argc==7)
    {
        std::string dir = string(argv[6]);    //data/output
        mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);    
        dir = string(argv[6]) + "/rgb/";
        mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        dir = string(argv[6]) + "/depth/";
        mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        dir = string(argv[6]) + "/mask/";
        mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    }

    
    /*
    S_IRWXU
    00700权限，代表该文件所有者拥有读，写和执行操作的权限
    
    S_IRUSR(S_IREAD)
    00400权限，代表该文件所有者拥有可读的权限
    
    S_IWUSR(S_IWRITE)
    00200权限，代表该文件所有者拥有可写的权限
    
    S_IXUSR(S_IEXEC)
    00100权限，代表该文件所有者拥有执行的权限
    
    S_IRWXG
    00070权限，代表该文件用户组拥有读，写和执行操作的权限
    
    S_IRGRP
    00040权限，代表该文件用户组拥有可读的权限
    
    S_IWGRP
    00020权限，代表该文件用户组拥有可写的权限
    
    S_IXGRP
    00010权限，代表该文件用户组拥有执行的权限
    
    S_IRWXO
    00007权限，代表其他用户拥有读，写和执行操作的权限
    
    S_IROTH
    00004权限，代表其他用户拥有可读的权限
    
    S_IWOTH
    00002权限，代表其他用户拥有可写的权限
    
    S_IXOTH
    00001权限，代表其他用户拥有执行的权限 
    
    */
    
    
    // Main loop
        cv::Mat imRGB, imD;
        cv::Mat imRGBOut, imDOut,maskOut;

      
        
        
    for(int ni=0; ni<nImages; ni++)
    {
        // Read image and depthmap from file
        imRGB = cv::imread(string(argv[3])+"/"+vstrImageFilenamesRGB[ni],CV_LOAD_IMAGE_UNCHANGED);   //以原始图像读取
        imD = cv::imread(string(argv[3])+"/"+vstrImageFilenamesD[ni],CV_LOAD_IMAGE_UNCHANGED);

        
        //imshow("123",imD);
        //cvWaitKey(0);
        
        
        
        
        double tframe = vTimestamps[ni];  //数据集中的时间节点

        if(imRGB.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenamesRGB[ni] << endl;
            return 1;
        }
        
        //argc == 7
        
        
        

#ifdef COMPILEDWITHC11    //c++11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        // Segment out the images
        cv::Mat mask = cv::Mat::ones(480,640,CV_8U);   //全部为1的矩阵
        if (argc == 6 || argc == 7)      //等于6 7 则用yolact  否则不用mask  
        {
            cv::Mat Yolact;
            
            //imshow("123",imRGB);   //imRGB图片传递没有问题
            //cvWaitKey(0);
            //cout<<argv[5]<<endl;     //data/mask
            //cout<<vstrImageFilenamesRGB[ni].replace(0,4,"")<<endl;   //1311870427.199132.png

            
            //string(argv[5]  data/mask
            //.replace(0,4,"")   去掉路径前缀  rgb
            
            Yolact = MaskNet->GetSegmentation(imRGB,string(argv[5]),vstrImageFilenamesRGB[ni].replace(0,4,""));
            
            //  Yolact 是mask
         
            
            
            cv::Mat Yolactdil = Yolact.clone();
            
            

            
            
            cv::dilate(Yolact,Yolactdil, kernel);  
            
            //膨胀操作  dilate函数使用像素领域内的局部极大运算符来膨胀一张图片
            //腐蚀操作  erode 函数使用像素领域内的局部极小云算法来腐蚀一张图片
            
            
            mask = mask - Yolactdil;
        }
        
         
        
        // Pass the image to the SLAM system
        if (argc == 7)        //7 保存结果  其他的不保存结果
        {
            SLAM.TrackRGBD(imRGB,imD,mask,tframe,imRGBOut,imDOut,maskOut);
        }
        else 
        {
            SLAM.TrackRGBD(imRGB,imD,mask,tframe);
        }
        

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif
        
        if (argc == 7)
        {
            cv::imwrite(string(argv[6]) + "/rgb/" + vstrImageFilenamesRGB[ni],imRGBOut);
            vstrImageFilenamesD[ni].replace(0,6,"");
            cv::imwrite(string(argv[6]) + "/depth/" + vstrImageFilenamesD[ni],imDOut);
            cv::imwrite(string(argv[6]) + "/mask/" + vstrImageFilenamesRGB[ni],maskOut);
        }
        
        imshow("mask",imRGBOut);
        cvWaitKey(1);


        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;   //代码计算一帧的时间

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)           
            T = vTimestamps[ni+1]-tframe;        //数据集中的时间节点+1 - 数据集中的时间节点   = 数据集一帧的时间
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];        //数据集中的时间节点 - 数据集中的时间节点-1   = 数据集一帧的时间

        if(ttrack<T)
            usleep((T-ttrack)*1e6);              //如果时间没到  就等着....     单位是微妙  10e-6
    }

    // Stop all threads


    while(SLAM.mpPointCloudMapping->loopbusy || SLAM.mpPointCloudMapping->cloudbusy)
    {
        cout<<"....";
    }

    SLAM.mpPointCloudMapping->bStop = true;

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());   
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    
    
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;

    // Save camera trajectory
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
    SLAM.SavePointCloud();

    SLAM.Shutdown();

    return 0;
}

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps)    //把时间节点 图片路径  深度图路径存到向量中去
{
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());          //./Examples/rgbd_dataset_freiburg2_desk_with_person/associations.txt
    while(!fAssociation.eof())
    {
        string s;
        getline(fAssociation,s);
        if(!s.empty())
        {
            stringstream ss;      
            ss << s;              //1311870427.199132 rgb/1311870427.199132.png 1311870427.207687 depth/1311870427.207687.png
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);

        }
    }
}

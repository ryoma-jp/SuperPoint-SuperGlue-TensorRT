//
// Created by haoyuefan on 2021/9/22.
//

#include <memory>
#include <chrono>
#include "utils.h"
#include "super_glue.h"
#include "super_point.h"

int main(int argc, char** argv){
    if (argc != 5) {
        std::cerr << "./superpointglue_sequence config_path model_dir image_folder_absolutely_path output_folder_path" << std::endl;
        return 0;
    }

    std::string config_path = argv[1];
    std::string model_dir = argv[2];
    std::string image_path = argv[3];
    std::string output_path = argv[4];
    std::vector<std::string> image_names;
    GetFileNames(image_path, image_names);
    Configs configs(config_path, model_dir);
    int width = configs.superglue_config.image_width;
    int height = configs.superglue_config.image_height;

    std::cout << "Building inference engine......" << std::endl;
    auto superpoint = std::make_shared<SuperPoint>(configs.superpoint_config);
    if (!superpoint->build()) {
        std::cerr << "Error in SuperPoint building engine. Please check your onnx model path." << std::endl;
        return 0;
    }
    auto superglue = std::make_shared<SuperGlue>(configs.superglue_config);
    if (!superglue->build()) {
        std::cerr << "Error in SuperGlue building engine. Please check your onnx model path." << std::endl;
        return 0;
    }
    std::cout << "SuperPoint and SuperGlue inference engine build success." << std::endl;

    Eigen::Matrix<double, 259, Eigen::Dynamic> feature_points0;
    cv::Mat image0 = cv::imread(image_names[0], cv::IMREAD_GRAYSCALE);
    if(image0.empty()) {
        std::cerr << "First image in the image folder is empty." << std::endl;
        return 0;
    }
    cv::resize(image0, image0, cv::Size(width, height));
    std::cout << "First image size: " << image0.cols << "x" << image0.rows << std::endl;
    if (!superpoint->infer(image0, feature_points0)) {
        std::cerr << "Failed when extracting features from first image." << std::endl;
        return 0;
    }
    std::vector<cv::DMatch> init_matches;
    superglue->matching_points(feature_points0, feature_points0, init_matches);
    std::string mkdir_cmd = "mkdir -p " + output_path;
    system(mkdir_cmd.c_str());

    std::ofstream log_file_all_feature_point(output_path + "/log_all_feature_point.csv");    // csv format: image index, keypoint index, feature point index, x, y, score
    log_file_all_feature_point<<"image_index,keypoint_index,feature_point_index,x,y,score"<<std::endl;
    std::ofstream log_file_match_points(output_path + "/log_match_points.csv");    // csv format: image index, keypoint0 feature point index, keypoint0 x, keypoint0 y, keypoint0 score, keypoint0 feature point index, keypoint1 x, keypoint1 y, keypoint1 score
    log_file_match_points<<"image_index,keypoint0_feature_point_index,keypoint0_x,keypoint0_y,keypoint0_score,keypoint1_feature_point_index,keypoint1_x,keypoint1_y,keypoint1_score"<<std::endl;
    for (int index = 1; index < image_names.size(); ++index) {
        Eigen::Matrix<double, 259, Eigen::Dynamic> feature_points1;
        std::vector<cv::DMatch> superglue_matches;
        cv::Mat image1 = cv::imread(image_names[index], cv::IMREAD_GRAYSCALE);
        if(image1.empty()) continue;
        cv::resize(image1, image1, cv::Size(width, height));

        std::cout << "Second image size: " << image1.cols << "x" << image1.rows << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        if (!superpoint->infer(image1, feature_points1)) {
            std::cerr << "Failed when extracting features from second image." << std::endl;
            return 0;
        }
        superglue->matching_points(feature_points0, feature_points1, superglue_matches);
        //std::cout << "Matches: " << std::to_string(superglue_matches.size()) << std::endl;
        //std::cout << "feature_points0: " << std::to_string(feature_points0.size()) << std::endl;
        if (superglue_matches.size() > 0) {
            for (auto &match : superglue_matches) {
                log_file_match_points<<index<<","
                                        <<match.queryIdx<<","
                                        <<feature_points0(1, match.queryIdx)<<","<<feature_points0(2, match.queryIdx)<<","<<feature_points0(0, match.queryIdx)<<","
                                        <<match.trainIdx<<","
                                        <<feature_points1(1, match.trainIdx)<<","<<feature_points1(2, match.trainIdx)<<","<<feature_points1(0, match.trainIdx)<<std::endl;
                //std::cout << "match: " << std::to_string(match.queryIdx) << ", " << std::to_string(match.trainIdx) << ", " << std::to_string(match.imgIdx) << std::endl;
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        cv::Mat match_image;
        std::vector<cv::KeyPoint> keypoints0, keypoints1;
        for (size_t i = 0; i < feature_points0.cols(); ++i) {
            double score = feature_points0(0, i);
            double x = feature_points0(1, i);
            double y = feature_points0(2, i);
            keypoints0.emplace_back(x, y, 8, -1, score);
            log_file_all_feature_point<<index<<","<<0<<","<<i<<","<<x<<","<<y<<","<<score<<std::endl;
        }
        for (size_t i = 0; i < feature_points1.cols(); ++i) {
            double score = feature_points1(0, i);
            double x = feature_points1(1, i);
            double y = feature_points1(2, i);
            keypoints1.emplace_back(x, y, 8, -1, score);
            log_file_all_feature_point<<index<<","<<1<<","<<i<<","<<x<<","<<y<<","<<score<<std::endl;
        }
        VisualizeMatching(image0, keypoints0, image1, keypoints1, superglue_matches, match_image, duration.count());
        cv::imwrite(output_path + "/" + std::to_string(index) + ".png", match_image);
    }
    log_file_all_feature_point.close();
    log_file_match_points.close();

  return 0;
}

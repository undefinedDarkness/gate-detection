#include "optional.hpp"
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <msd/channel.hpp>
#include <opencv2/opencv.hpp>
#include <thread>
#include <unordered_set>
#include <vector>

float degrees(float radians) { return radians * 180 / CV_PI; }

float radians(float degrees) { return degrees * CV_PI / 180; }

using namespace std;
using namespace cv;

Mat camMatrix, distCoef;
double FOCAL_LENGTH_PIXELS, IMAGE_WIDTH, HFOV, FOCAL_LENGTH_METERS,
    SENSOR_WIDTH, meters_to_pixels, pixels_to_meters;

Vec3d rotation_matrix_to_euler_angles(Mat rotation_matrix) {
  double sy =
      sqrt(rotation_matrix.at<double>(0, 0) * rotation_matrix.at<double>(0, 0) +
           rotation_matrix.at<double>(1, 0) * rotation_matrix.at<double>(1, 0));

  bool singular = sy < 1e-6;

  double x, y, z;
  if (!singular) {
    x = atan2(rotation_matrix.at<double>(2, 1),
              rotation_matrix.at<double>(2, 2));
    y = atan2(-rotation_matrix.at<double>(2, 0), sy);
    z = atan2(rotation_matrix.at<double>(1, 0),
              rotation_matrix.at<double>(0, 0));
  } else {
    x = atan2(-rotation_matrix.at<double>(1, 2),
              rotation_matrix.at<double>(1, 1));
    y = atan2(-rotation_matrix.at<double>(2, 0), sy);
    z = 0;
  }

  return Vec3d(x * 180 / CV_PI, y * 180 / CV_PI, z * 180 / CV_PI);
}

struct LineInfo {
  Vec4i line;
  Vec2f direction;
};

struct PoseInfo {
  double roll;
  double pitch;
  double yaw;
  double distance;
};

class ImageFilters {
private:
  Ptr<CLAHE> clahe_for_l = createCLAHE(4.0, Size(16, 16));
  Ptr<CLAHE> clahe_for_a = createCLAHE(2.0, Size(4, 4));
  Ptr<CLAHE> clahe_for_b = createCLAHE(2.0, Size(4, 4));
  Mat lab_image;
  vector<Mat> lab_channels;
  vector<Point2f> intersections;
  vector<Point2f> clusters;
  unordered_set<int> used_points;
  vector<Point2f> cluster;

  Mat applyCLAHE(const Mat &image) {
    cvtColor(image, lab_image, COLOR_BGR2Lab);
    split(lab_image, lab_channels);

#pragma omp parallel sections
    {
#pragma omp section
      { clahe_for_l->apply(lab_channels[0], lab_channels[0]); }
#pragma omp section
      { clahe_for_a->apply(lab_channels[1], lab_channels[1]); }
#pragma omp section
      { clahe_for_b->apply(lab_channels[2], lab_channels[2]); }
    }

    merge(lab_channels, lab_image);
    cvtColor(lab_image, lab_image, COLOR_Lab2BGR);
    return lab_image;
  }

  // Can be fully moved to CUDA
  Mat applyCLAHE_CUDA(const Mat &image) {
    cvtColor(image, lab_image, COLOR_BGR2Lab);
    split(lab_image, lab_channels);

#pragma omp parallel sections
    {
#pragma omp section
      { clahe_for_l->apply(lab_channels[0], lab_channels[0]); }
#pragma omp section
      { clahe_for_a->apply(lab_channels[1], lab_channels[1]); }
#pragma omp section
      { clahe_for_b->apply(lab_channels[2], lab_channels[2]); }
    }

    merge(lab_channels, lab_image);
    cvtColor(lab_image, lab_image, COLOR_Lab2BGR);
    return lab_image;
  }

  Mat adjustExposure(const Mat &image, double brightness) {
    double exposure_factor = (-0.0044117) * brightness + 1.695287;
    Mat balanced_image;
    // Use .convertTo in CUDA
    convertScaleAbs(image * exposure_factor, balanced_image, 1, 0);
    return balanced_image;
  }

  Mat adjustExposure_CUDA(const Mat &image, double brightness) {
    double exposure_factor = (-0.0044117) * brightness + 1.695287;
    Mat balanced_image;
    // Use .convertTo in CUDA
    convertScaleAbs(image * exposure_factor, balanced_image, 1, 0);
    return balanced_image;
  }

  Mat applySobelAndThreshold(const Mat &image) {
    Mat blurred_image, gray_image;
    GaussianBlur(image, blurred_image, Size(3, 3), 0);
    cvtColor(blurred_image, gray_image, COLOR_BGR2GRAY);

    Mat grad_x, grad_y, abs_grad_x, abs_grad_y, grad;
    double scale = 1.2;
    Sobel(gray_image, grad_x, CV_16S, 1, 0, 3, scale, 0, BORDER_DEFAULT);
    Sobel(gray_image, grad_y, CV_16S, 0, 1, 3, scale, 0, BORDER_DEFAULT);

    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

    threshold(grad, grad, 50, 255, THRESH_BINARY);
    return grad;
  }

  Mat applySobelAndThreshold_CUDA(const Mat &image) {
    // Use GaussianFilter and SobelFilter in CUDA
    Mat blurred_image, gray_image;
    GaussianBlur(image, blurred_image, Size(3, 3), 0);
    cvtColor(blurred_image, gray_image, COLOR_BGR2GRAY);

    Mat grad_x, grad_y, abs_grad_x, abs_grad_y, grad;
    double scale = 1.2;
    Sobel(gray_image, grad_x, CV_16S, 1, 0, 3, scale, 0, BORDER_DEFAULT);
    Sobel(gray_image, grad_y, CV_16S, 0, 1, 3, scale, 0, BORDER_DEFAULT);

    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

    threshold(grad, grad, 50, 255, THRESH_BINARY);
    return grad;
  }

  vector<LineInfo> extended_lines;
  vector<Vec4i> lines;
  vector<LineInfo> detectAndExtendLines(const Mat &grad) {
    // HoughSegmentDetector
    lines.clear();
    extended_lines.clear();
    HoughLinesP(grad, lines, 1, CV_PI / 180, 50, 75, 10);

    // BREAK FROM GPU PROCESSING HERE

    for (size_t i = 0; i < lines.size(); i++) {
      const Vec4i &l = lines[i];
      double dx = l[2] - l[0];
      double dy = l[3] - l[1];
      double length = sqrt(dx * dx + dy * dy);

      Vec2f direction(dx / length, dy / length);

      double extend_length = 100;
      int new_x1 = static_cast<int>(l[0] - direction[0] * extend_length);
      int new_y1 = static_cast<int>(l[1] - direction[1] * extend_length);
      int new_x2 = static_cast<int>(l[2] + direction[0] * extend_length);
      int new_y2 = static_cast<int>(l[3] + direction[1] * extend_length);

      extended_lines.push_back(
          {Vec4i(new_x1, new_y1, new_x2, new_y2), direction});
    }
    return extended_lines;
  }

  double calculateAngle(const Vec2f &dir1, const Vec2f &dir2) {
    double dot_product = dir1[0] * dir2[0] + dir1[1] * dir2[1];
    double angle = acos(dot_product) * 180.0 / CV_PI;
    return angle;
  }

  vector<Point2f> findIntersections(const vector<LineInfo> &lines) {
    intersections.clear();

    // Can be written as a custom CUDA kernel

#pragma omp simd
    for (size_t i = 0; i < lines.size(); i++) {
      for (size_t j = i + 1; j < lines.size(); j++) {
        const Vec4i &l1 = lines[i].line;
        const Vec4i &l2 = lines[j].line;

        double a1 = l1[3] - l1[1];
        double b1 = l1[0] - l1[2];
        double c1 = a1 * l1[0] + b1 * l1[1];

        double a2 = l2[3] - l2[1];
        double b2 = l2[0] - l2[2];
        double c2 = a2 * l2[0] + b2 * l2[1];

        double det = a1 * b2 - a2 * b1;

        if (det != 0) {
          double angle = calculateAngle(lines[i].direction, lines[j].direction);

          if (70 < angle && angle < 110) {
            float x = static_cast<float>((b2 * c1 - b1 * c2) / det);
            float y = static_cast<float>((a1 * c2 - a2 * c1) / det);
            intersections.emplace_back(x, y);
          }
        }
      }
    }
    return intersections;
  }

  vector<Point2f> clusterIntersections(const vector<Point2f> &intersections) {
    clusters.clear();
    used_points.clear();
    const int cluster_threshold = 2500; // 50 squared
    const int min_cluster_size = 20;

    for (size_t i = 0; i < intersections.size(); i++) {
      if (used_points.find(i) != used_points.end())
        continue;

      cluster.clear();
      cluster.push_back(intersections[i]);
      used_points.insert(i);

      for (size_t j = i + 1; j < intersections.size(); j++) {
        if (used_points.count(j))
          continue;

        float dx = intersections[i].x - intersections[j].x;
        float dy = intersections[i].y - intersections[j].y;
        if (dx * dx + dy * dy < cluster_threshold) {
          cluster.push_back(intersections[j]);
          used_points.insert(j);
        }
      }

      if (cluster.size() > min_cluster_size) {
        float avg_x = 0, avg_y = 0;
        for (const auto &p : cluster) {
          avg_x += p.x;
          avg_y += p.y;
        }
        clusters.emplace_back(avg_x / cluster.size(), avg_y / cluster.size());
      }
    }
    return clusters;
  }

  void drawLines(Mat &image, const vector<LineInfo> &lines) {
    for (const auto &line_info : lines) {
      const Vec4i &l = line_info.line;
      line(image, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3,
           LINE_AA);
    }
  }

  void drawClusters(Mat &image, const vector<Point2f> &clusters) {
    for (const auto &point : clusters) {
      circle(image, point, 5, Scalar(0, 255, 255), -1);
    }
  }

  tl::optional<std::pair<vector<Point2f>, PoseInfo>>
  estimate_pose(const vector<Point2f> &points) {
    if (points.size() != 4) {
      cout << "4 points required" << endl;
      return tl::nullopt;
    }

    double rectangle_length_m = 1.4;
    double rectangle_breadth_m = 1.0;

    double rectangle_length_px = rectangle_length_m * meters_to_pixels;
    double rectangle_breadth_px = rectangle_breadth_m * meters_to_pixels;

    vector<Point3f> model_points = {
        Point3f(0.0, 0.0, 0.0), Point3f(rectangle_length_px, 0.0, 0.0),
        Point3f(rectangle_length_px, rectangle_breadth_px, 0.0),
        Point3f(0.0, rectangle_breadth_px, 0.0)};

    Mat rotation_vector, translation_vector;
    bool success = solvePnP(model_points, points, camMatrix, distCoef,
                            rotation_vector, translation_vector);

    if (!success) {
      cout << "Pose estimation failed" << endl;
      return tl::nullopt;
    }

    Mat rotation_matrix;
    Rodrigues(rotation_vector, rotation_matrix);

    Vec3d euler_angles = rotation_matrix_to_euler_angles(rotation_matrix);
    double roll = euler_angles[0], pitch = euler_angles[1],
           yaw = euler_angles[2];

    double distance_in_pixels = norm(translation_vector);
    double distance_in_meters = distance_in_pixels * pixels_to_meters;

    if (yaw < -90) {
      yaw = -(180 + yaw);
    } else {
      yaw = (180 - yaw);
    }

    cout << "Pose estimation successful!" << endl;
    cout << "Roll: " << roll << " degrees" << endl;
    cout << "Pitch: " << pitch << " degrees" << endl;
    cout << "Yaw: " << yaw << " degrees" << endl;
    cout << "Distance: " << distance_in_meters << " meters" << endl;

    return tl::optional<std::pair<vector<Point2f>, PoseInfo>>(
        {points, {roll, pitch, yaw, distance_in_meters}});
  }

public:
  ImageFilters() {
    clusters.reserve(1024);
    used_points.reserve(1024);
    intersections.reserve(1024);
    cluster.reserve(1024);
    extended_lines.reserve(1024);
    lines.reserve(1024);
  }

  Mat Filters(const Mat &image) {
    Mat image_copy2 = image.clone();

    // Apply CLAHE
    Mat image_clahe = applyCLAHE(image);

    // Adjust exposure
    Mat gray_image;
    cvtColor(image, gray_image, COLOR_BGR2GRAY);
    double brightness = mean(gray_image)[0];
    Mat balanced_image = adjustExposure(image_clahe, brightness);

    // Apply Sobel and threshold
    Mat grad = applySobelAndThreshold(balanced_image);

    // Detect and extend lines
    vector<LineInfo> extended_lines = detectAndExtendLines(grad);

    // Draw the extended lines
    drawLines(image_copy2, extended_lines);

    // Find intersections and cluster them
    vector<Point2f> intersections = findIntersections(extended_lines);
    vector<Point2f> clusters = clusterIntersections(intersections);

    // Draw clusters
    drawClusters(image_copy2, clusters);

    // Estimate pose if we have enough clusters
    if (clusters.size() == 4) {
      auto res = estimate_pose(clusters);
      if (res) {
        // Draw roll, pitch, yaw, and distance on the image
        stringstream ss;
        ss << "Roll: " << res->second.roll << " degrees";
        putText(image_copy2, ss.str(), Point(10, 30), FONT_HERSHEY_SIMPLEX, 1,
                Scalar(0, 0, 0), 2);
        ss.str("");
        ss << "Pitch: " << res->second.pitch << " degrees";
        putText(image_copy2, ss.str(), Point(10, 60), FONT_HERSHEY_SIMPLEX, 1,
                Scalar(0, 0, 0), 2);
        ss.str("");
        ss << "Yaw: " << res->second.yaw << " degrees";
        putText(image_copy2, ss.str(), Point(10, 90), FONT_HERSHEY_SIMPLEX, 1,
                Scalar(0, 0, 0), 2);
        ss.str("");
        ss << "Distance: " << res->second.distance << " meters";
        putText(image_copy2, ss.str(), Point(10, 120), FONT_HERSHEY_SIMPLEX, 1,
                Scalar(0, 0, 0), 2);
      }
    }

    return image_copy2;
  }
};

struct InputFrame {
  Mat image;
  std::chrono::high_resolution_clock::time_point timestamp;
  int index;
};

struct OutputFrame {
  Mat image;
  int index;
  int delay;
};

class ImageProcessor {

  msd::channel<InputFrame> &input_channel;
  msd::channel<OutputFrame> &output_channel;

  ImageFilters filters;
  stringstream ss;
  int processorId;
  bool isRunning = true;

  void processFrame(InputFrame &frame) {
    // cout << processorId << " Got frame: " << frame.index << endl;
    Mat filtered_image = filters.Filters(frame.image);
    auto end = chrono::high_resolution_clock::now();
    int duration =
        chrono::duration_cast<chrono::milliseconds>(end - frame.timestamp)
            .count();

    // Draw duration and frame index on the image
    ss.str("");
    ss << "Index: " << frame.index;
    putText(filtered_image, ss.str(), Point(10, 180), FONT_HERSHEY_SIMPLEX, 1,
            Scalar(0, 0, 0), 2);

    hconcat(frame.image, filtered_image, filtered_image);

    output_channel << OutputFrame({filtered_image, frame.index, duration});
  }

public:
  ImageProcessor(msd::channel<InputFrame> &input_channel,
                 msd::channel<OutputFrame> &output_channel, int processorId)
      : input_channel(input_channel), output_channel(output_channel),
        processorId(processorId) {}

  void stop() { isRunning = false; }

  void run() {
    while (isRunning) {
      InputFrame frame;
      input_channel >> frame;
      processFrame(frame);
    }
  }
};

int initParameters() {
  // Load camera calibration parameters
  try {
    FileStorage fs("calibration_params.yml", FileStorage::READ);
    if (!fs.isOpened()) {
      cout << "Could not open the calibration file: calibration_params.yml"
           << endl;
      return -1;
    }
    fs["camMatrix"] >> camMatrix;
    fs["distCoef"] >> distCoef;
    fs.release();
  } catch (const cv::Exception &e) {
    cerr << "GOT ERROR LOADING CAMERA MATRIX" << endl;
    cerr << e.what() << endl;
    return -1;
  }

  // exit(0);

  std::cout << "Loaded Camera Matrix\n";

  // Calculate camera parameters
  FOCAL_LENGTH_PIXELS = camMatrix.at<double>(0, 0);
  IMAGE_WIDTH = camMatrix.at<double>(0, 2) * 2;
  HFOV =
      2 *
      degrees(atan(
          IMAGE_WIDTH /
          (2 * FOCAL_LENGTH_PIXELS))); // 2 * atan((IMAGE_WIDTH / (2 *
                                       // FOCAL_LENGTH_PIXELS))) * 180 / CV_PI;
  FOCAL_LENGTH_METERS = FOCAL_LENGTH_PIXELS / IMAGE_WIDTH;
  SENSOR_WIDTH = 2 * FOCAL_LENGTH_METERS * tan(radians(HFOV / 2));
  meters_to_pixels = FOCAL_LENGTH_PIXELS / SENSOR_WIDTH;
  pixels_to_meters = 1 / meters_to_pixels;
  return 0;
}

#define NPROCESSORS 2

int main() {
  std::cout << "ASAD's Code C++ VERSION\n";

  if (initParameters() == -1) {
    return -1;
  }

  string image_folder = "./dataset/Half_time/";
  vector<string> image_paths;
  glob(image_folder + "*.png", image_paths, false);

  if (image_paths.empty()) {
    cout << "No images found in folder: " << image_folder << endl;
    return 1;
  }

  namedWindow("Original and Filtered Footage", WINDOW_NORMAL);

  msd::channel<InputFrame> input_channel{5};
  msd::channel<OutputFrame> output_channel;

  vector<std::pair<ImageProcessor, thread>> processors;

  for (int i = 0; i < NPROCESSORS; i++) {
    processors.emplace_back(ImageProcessor(input_channel, output_channel, i),
                            thread(&ImageProcessor::run, &processors[i].first));
  }

  bool imageReaderRunning = true;
  thread image_reader_thread([&]() {
    for (size_t i = 0; i < image_paths.size(); ++i) {
      if (!imageReaderRunning) {
        break;
      }
      Mat image = imread(image_paths[i]);
      if (image.empty()) {
        cout << "Could not read image: " << image_paths[i] << endl;
        continue;
      }
      auto timestamp = chrono::high_resolution_clock::now();
      input_channel << InputFrame{std::move(image), timestamp,
                                  static_cast<int>(i)};
      // this_thread::sleep_for(chrono::milliseconds(100)); // Simulate delay
    }
  });

  // thread image_display_thread([&]() {
  auto start = chrono::high_resolution_clock::now();
  stringstream ss;
  while (true) {
    OutputFrame output_frame;
    output_channel >> output_frame;

    auto end = chrono::high_resolution_clock::now();
    auto duration =
        chrono::duration_cast<chrono::milliseconds>(end - start).count();
    start = end;
    double fps = duration == 0 ? -1.0 : 1.0 / duration;

    ss.str("");
    ss << "FPS: " << fps * 1000.0;
    putText(output_frame.image, ss.str(), Point(10, 150), FONT_HERSHEY_SIMPLEX,
            1, Scalar(0, 0, 0), 2);

    imshow("Original and Filtered Footage", output_frame.image);
    if (waitKey(1) == 27) { // Exit on ESC key
      break;
    }
  }
  // });

  for (auto &processor : processors) {
    processor.first.stop();
    processor.second.join();
  }

  imageReaderRunning = false;

  std::cout << "Finished\n";
  return 0;
}

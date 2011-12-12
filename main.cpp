#include <sim/sim.hpp>
#include <boost/thread.hpp>
#include <opencv2/highgui/highgui.hpp>
int
main(int argc, char *argv[])
{
  using namespace sim;
  // Parse command line arguments
  if (argc != 2)
  {
    std::cerr << "Usage: " << argv[0] << " Filename.png" << std::endl;
    return EXIT_FAILURE;
  }

  std::string inputFilename = argv[1];

  SimRunner sm(inputFilename, 0.8, 0.4, 640, 480);
  boost::thread t(boost::ref(sm));
  cv::Mat image, depth, mask, R, T, K;
  bool new_data;
  while (!sm.get_data(image, depth, mask, K, R, T, new_data))
  {
    if (new_data)
    {
      cv::imshow("Render", image);
      std::cout << "K = " << K << "\nR = " << R << "\nT = " << T << std::endl;
    }
    int key = 0xFF & cv::waitKey(10);
    if (key == 'q')
    {
      sm.quit();
    }
    boost::this_thread::sleep(boost::posix_time::milliseconds(10));
  }
  t.join();
  std::cout << "exiting" << std::endl;
  return EXIT_SUCCESS;
}

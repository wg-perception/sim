#include <ecto/ecto.hpp>
#include <sim/sim.hpp>
namespace sim
{
  using ecto::tendrils;
  struct PlanarSim
  {
    static
    void
    declare_params(tendrils& p)
    {
      p.declare<std::string>("image_name", "Image file name, should be a 3 channel png.").required(true);
      p.declare<double>("width", "width in meters.").required(true);
      p.declare<double>("height", "height in meters.").required(true);
      p.declare<int>("window_width", "Window width in pixels", 640);
      p.declare<int>("window_height", "Window height in pixels", 480);
    }

    static
    void
    declare_io(const tendrils& p, tendrils& i, tendrils& o)
    {
      o.declare<cv::Mat>("image", "Image generated");
      o.declare<cv::Mat>("depth", "Depth generated");
      o.declare<cv::Mat>("mask", "Valid depth points");

      o.declare<cv::Mat>("R", "Rotation matrix, double, 3x3");
      o.declare<cv::Mat>("T", "Translation matrix, double, 3x1");
      o.declare<cv::Mat>("K", "Camera Intrisics matrix, double, 3x3");
    }

    void
    configure(const tendrils& p, const tendrils& i, const tendrils& o)
    {
      double width, height;
      int window_width, window_height;
      std::string image_name;
      p["image_name"] >> image_name;
      p["window_width"] >> window_width;
      p["window_height"] >> window_height;
      p["width"] >> width;
      p["height"] >> height;
      sm.reset(new SimRunner(image_name, width, height, window_width, window_height));
      t.reset(new boost::thread(boost::ref(*sm)));
      image = o["image"];
      depth = o["depth"];
      mask = o["mask"];
      R = o["R"];
      K = o["K"];
      T = o["T"];
    }

    int
    process(const tendrils& i, const tendrils& o)
    {
      bool new_data;
      if (sm->get_data(*image, *depth, *mask, *K, *R, *T, new_data))
      {
        t->join();
        return ecto::QUIT;
      }
      return ecto::OK;
    }
    ~PlanarSim()
    {
      if (sm)
      {
        sm->quit();
      }
      if (t)
      {
        t->interrupt();
        t->join();
      }
    }
    boost::shared_ptr<SimRunner> sm;
    boost::shared_ptr<boost::thread> t;
    ecto::spore<cv::Mat> image, depth, mask, R, T, K;
  };
}

ECTO_CELL(sim, sim::PlanarSim, "PlanarSim", "Simulates a view of a planar object.");

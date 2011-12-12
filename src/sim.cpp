#include <sim/sim.hpp>

#include <vtkImageData.h>
#include <vtkPNGReader.h>
#include <vtkPolyDataMapper.h>
#include <vtkSmartPointer.h>
#include <vtkTextureMapToPlane.h>
#include <vtkPlaneSource.h>
#include <vtkTexture.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkCommand.h>
#include <vtkCallbackCommand.h>
#include <vtkCamera.h>
#include <vtkMatrix4x4.h>
#include <vtkRendererCollection.h>

#include <vtkOBJReader.h>

#include <boost/thread.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#define SP(X) vtkSmartPointer< X >

namespace sim
{
  void
  vtk_to_K(cv::Size sz, vtkCamera* cam, cv::Mat& K, cv::Mat& R, cv::Mat& T);

  void
  grab_frame(vtkRenderWindow* renderWindow, cv::Mat& image, cv::Mat& depth, cv::Mat& mask, cv::Mat& K, cv::Mat& R,
             cv::Mat& T);
  void
  grab_frame_callback(vtkObject* caller, long unsigned int eventId, void* clientData, void* callData);

  void
  quit_callback(vtkObject* caller, long unsigned int eventId, void* clientData, void* callData);

  struct SimRunner::Impl
  {

    Impl(const std::string image_name, double width, double height, int window_width, int window_height)
        :
          data_ready(false),
          has_quit(false),
          quit(false)
    {
      // Read the image which will be the texture
      vtkSmartPointer<vtkPNGReader> pngReader = vtkSmartPointer<vtkPNGReader>::New();
      pngReader->SetFileName(image_name.c_str());

      // Create a plane
      vtkSmartPointer<vtkPlaneSource> plane = vtkSmartPointer<vtkPlaneSource>::New();
      plane->SetCenter(0, 0, 0);
      plane->SetOrigin(-width / 2, -height / 2, 0.000000);
      plane->SetPoint1(width / 2, -height / 2, 0.000000);
      plane->SetPoint2(-width / 2, height / 2, 0.000000);
      plane->SetNormal(0, 0, 1);
      // Apply the texture
      vtkSmartPointer<vtkTexture> texture = vtkSmartPointer<vtkTexture>::New();
      texture->SetInput(pngReader->GetOutput());

      vtkSmartPointer<vtkTextureMapToPlane> texturePlane = vtkSmartPointer<vtkTextureMapToPlane>::New();
      texturePlane->SetInput(plane->GetOutput());

      vtkSmartPointer<vtkPolyDataMapper> planeMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
      planeMapper->SetInputConnection(texturePlane->GetOutputPort());

      vtkSmartPointer<vtkActor> texturedPlane = vtkSmartPointer<vtkActor>::New();
      texturedPlane->SetMapper(planeMapper);
      texturedPlane->SetTexture(texture);

      // Visualize the textured plane
      renderer = vtkSmartPointer<vtkRenderer>::New();
      renderer->AddActor(texturedPlane);

      renderer->SetAmbient(1, 1, 1);
      renderer->SetBackground(0.2, 0.2, 0.2); // Background color
      renderer->ResetCamera();

      renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
      renderWindow->SetSize(window_width, window_height);
      renderWindow->AddRenderer(renderer);

      {
        vtkSmartPointer<vtkOBJReader> obj_reader = vtkSmartPointer<vtkOBJReader>::New();
        obj_reader->SetFileName("/home/erublee/recognition_kitchen/sim/art/box.uv.obj");
        obj_reader->Update();

        vtkSmartPointer<vtkPolyDataMapper> objMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        objMapper->SetInput(obj_reader->GetOutput());
        //objMapper->SetScalarMaterialModeToAmbient();
        vtkSmartPointer<vtkPNGReader> pngReader = vtkSmartPointer<vtkPNGReader>::New();
        pngReader->SetFileName("/home/erublee/recognition_kitchen/sim/art/box.png");
        vtkSmartPointer<vtkTexture> texture = vtkSmartPointer<vtkTexture>::New();
        texture->SetInput(pngReader->GetOutput());

        vtkSmartPointer<vtkActor> objActor = vtkSmartPointer<vtkActor>::New();
        objActor->SetMapper(objMapper);
        objActor->SetTexture(texture);
        renderer->AddActor(objActor);
      }

      renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();
      renderWindowInteractor->SetRenderWindow(renderWindow);

      //track ball interaction
      style = vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New();
      renderWindowInteractor->SetInteractorStyle(style);
      renderer->GetActiveCamera()->SetViewAngle(45); //45 degree viewing angle.
    }

    void
    operator()()
    {
      renderWindowInteractor->Initialize();
      vtkSmartPointer<vtkCallbackCommand> quitcb = vtkSmartPointer<vtkCallbackCommand>::New();
      quitcb->SetCallback(quit_callback);
      quitcb->SetClientData(this);
      renderWindowInteractor->CreateRepeatingTimer(100);
      renderWindowInteractor->AddObserver(vtkCommand::TimerEvent, quitcb);

      cb = vtkSmartPointer<vtkCallbackCommand>::New();
      cb->SetCallback(grab_frame_callback);
      cb->SetClientData(this);
      renderWindowInteractor->AddObserver(vtkCommand::RenderEvent, cb);
      renderWindowInteractor->Render();
      renderWindowInteractor->Start();
      has_quit = true;
    }

    void
    put_data(const cv::Mat& image, const cv::Mat& depth, const cv::Mat& mask, const cv::Mat&K, const cv::Mat& R,
             const cv::Mat& T)
    {
      {
        boost::lock_guard<boost::mutex> lock(mtx);
        this->image = image;
        this->depth = depth;
        this->mask = mask;
        this->K = K;
        this->R = R;
        this->T = T;
        data_ready = true;
      }
      condition.notify_one();
    }
    bool
    get_data(cv::Mat& image, cv::Mat& depth, cv::Mat& mask, cv::Mat&K, cv::Mat& R, cv::Mat& T, bool& new_data)
    {
      boost::unique_lock<boost::mutex> lock(mtx);
      while (!data_ready)
      {
        condition.timed_wait(lock, boost::posix_time::milliseconds(33));
        break;
      }
      new_data = data_ready;
      if (new_data)
      {
        image = this->image;
        depth = this->depth;
        mask = this->mask;
        K = this->K;
        R = this->R;
        T = this->T;
      }
      data_ready = false;
      return has_quit;
    }
    cv::Mat image, depth, mask, R, T, K;
    bool data_ready, has_quit;

    boost::condition_variable condition;
    boost::mutex mtx;

    // Visualize the textured plane
    vtkSmartPointer<vtkRenderer> renderer;
    vtkSmartPointer<vtkRenderWindow> renderWindow;
    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor;

    //key callbacks
    vtkSmartPointer<vtkCallbackCommand> cb;
    //track ball interaction
    vtkSmartPointer<vtkInteractorStyleTrackballCamera> style;

    bool quit;
  };

  void
  quit_callback(vtkObject* caller, long unsigned int eventId, void* clientData, void* callData)
  {
    vtkRenderWindowInteractor *iren = static_cast<vtkRenderWindowInteractor*>(caller);
    if (static_cast<SimRunner::Impl*>(clientData)->quit)
    {
      iren->InvokeEvent(vtkCommand::ExitEvent);
    }
  }

  void
  grab_frame_callback(vtkObject* caller, long unsigned int eventId, void* clientData, void* callData)
  {
    vtkRenderWindowInteractor *iren = static_cast<vtkRenderWindowInteractor*>(caller);
    cv::Mat image, depth, mask, R, T, K;
    grab_frame(iren->GetRenderWindow(), image, depth, mask, K, R, T);
    static_cast<SimRunner::Impl*>(clientData)->put_data(image, depth, mask, K, R, T);
  }

  void
  view_angles(double degrees, double aspect_ratio, double& alpha, double& beta)
  {
    alpha = (degrees * CV_PI / 180.0) / 2.0; //in radians, half the angle.
    beta = atan(tan(alpha) * aspect_ratio); // assume that the angles are related such that the focal length must be the same.
  }

  double
  f_from_fov(double alpha, double w)
  {
    return w / tan(alpha);
  }

  void
  vtk_to_K(cv::Size sz, vtkCamera* cam, cv::Mat& K, cv::Mat& R, cv::Mat& T)
  {

    K = cv::Mat::eye(3, 3, CV_64F);
    double cx = (sz.width - 1.0) / 2.0;
    double cy = (sz.height - 1.0) / 2.0;
    double fovx, fovy;
    double f;
    if (cam->GetUseHorizontalViewAngle())
    {
      //horizonal view anble
      view_angles(cam->GetViewAngle(), cy / cx, fovx, fovy);
    }
    else
    {
      //vertical view angle
      view_angles(cam->GetViewAngle(), cx / cy, fovy, fovx);
    }
    //calculate our focal length based of the fov in X and our center x
    f = f_from_fov(fovx, cx);

    K.at<double>(0, 0) = f;
    K.at<double>(0, 2) = cx;
    K.at<double>(1, 1) = f;
    K.at<double>(1, 2) = cy;

    double x, y, z;
    // construct a rotation matrix around x.
    cv::Mat Rx = cv::Mat::eye(3, 3, CV_64F);
    Rx.at<double>(1, 1) = -1;
    Rx.at<double>(2, 2) = -1;
    cv::Mat Rz = cv::Mat::eye(3, 3, CV_64F);
    Rz.at<double>(0, 0) = 0;
    Rz.at<double>(1, 1) = 0;
    Rz.at<double>(0, 1) = 1;
    Rz.at<double>(1, 0) = -1;
    cam->GetPosition(x, y, z);
    vtkMatrix4x4* vm = cam->GetViewTransformMatrix();
    cv::Mat tR(4, 4, CV_64F, vm->Element);
    {
      cv::Mat sub(tR(cv::Range(0, 3), cv::Range(0, 3)));
      sub.copyTo(R);
      R = Rx * R;
    }
    {
      cv::Mat sub(tR(cv::Range(0, 3), cv::Range(3, 4)));
      sub.copyTo(T);
      T = Rx * T;
    }
  }

  void
  grab_frame(vtkRenderWindow* renderWindow, cv::Mat& image, cv::Mat& depth, cv::Mat& mask, cv::Mat& K, cv::Mat& R,
             cv::Mat& T)
  {
    int * ws = renderWindow->GetSize();
    cv::Size sz(ws[0], ws[1]);
    vtkCamera* camera = renderWindow->GetRenderers()->GetFirstRenderer()->GetActiveCamera();
    vtk_to_K(sz, camera, K, R, T);
    {
      unsigned char* pixel_data = renderWindow->GetRGBACharPixelData(0, 0, sz.width - 1, sz.height - 1, 1);
      cv::cvtColor(cv::Mat(sz, CV_8UC4, pixel_data), image, CV_RGBA2BGR);
      delete[] pixel_data; //need to delete this buffer
    }

    {
      double near, far;
      camera->GetClippingRange(near, far);
      float * zbuffer = renderWindow->GetZbufferData(0, 0, sz.width - 1, sz.height - 1);
      cv::Mat zt(sz, CV_32F, zbuffer);
      zt.convertTo(depth, CV_32F, far, near); //rescale , 0 is near, 1 is far
      mask = zt == 1.0;
      depth.setTo(cv::Scalar::all(0), mask); //mask off and set all far to some magic number, for now 0
      mask = mask != 0;
      delete[] zbuffer; //need to delete this buffer
    }
    cv::flip(image, image, 0); //vertical flip.
    cv::flip(depth, depth, 0); //vertical flip.
  }

  SimRunner::SimRunner(const std::string image_name, double width, double height, int window_width, int window_height)
      :
        impl_(new  Impl(image_name, width, height, window_width, window_height))
  {
  }

  void
  SimRunner::operator ()()
  {
    (*impl_)();
  }

  bool
  SimRunner::get_data(cv::Mat & image, cv::Mat & depth, cv::Mat & mask, cv::Mat & K, cv::Mat & R, cv::Mat & T,
                      bool & new_data)
  {
    return impl_->get_data(image, depth, mask, K, R, T, new_data);
  }

  void SimRunner::quit()
  {
    impl_->quit = true;
}


}

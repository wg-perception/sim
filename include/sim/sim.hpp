/*
 * sim.hpp
 *
 *  Created on: Dec 11, 2011
 *      Author: erublee
 */
#pragma once
#ifndef SIM_HPP_KDJFIUFIUDKLJ89078760DFDF
#define SIM_HPP_KDJFIUFIUDKLJ89078760DFDF

#include <opencv2/core/core.hpp>
#include <boost/shared_ptr.hpp>
namespace sim
{

  struct SimRunner
  {

    SimRunner(const std::string image_name, double width, double height, int window_width, int window_height);

    void
    operator()();

    bool
    get_data(cv::Mat& image, cv::Mat& depth, cv::Mat& mask, cv::Mat&K, cv::Mat& R, cv::Mat& T, bool& new_data);

    void quit();

    struct Impl;
    boost::shared_ptr<Impl> impl_;
  };
}

#endif /* SIM_HPP_KDJFIUFIUDKLJ89078760DFDF */

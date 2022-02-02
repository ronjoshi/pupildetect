#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

//#include <mgl2/mgl.h>

#include <iostream>
#include <queue>
#include <stdio.h>

#include "constants.h"
#include "helpers.h"

// Pre-declarations
cv::Mat floodKillEdges(cv::Mat &mat);

#pragma mark Visualization
/*
template<typename T> mglData *matToData(const cv::Mat &mat) {
  mglData *data = new mglData(mat.cols,mat.rows);
  for (int y = 0; y < mat.rows; ++y) {
    const T *Mr = mat.ptr<T>(y);
    for (int x = 0; x < mat.cols; ++x) {
      data->Put(((mreal)Mr[x]),x,y);
    }
  }
  return data;
}

void plotVecField(const cv::Mat &gradientX, const cv::Mat &gradientY, const cv::Mat &img) {
  mglData *xData = matToData<double>(gradientX);
  mglData *yData = matToData<double>(gradientY);
  mglData *imgData = matToData<float>(img);
  
  mglGraph gr(0,gradientX.cols * 20, gradientY.rows * 20);
  gr.Vect(*xData, *yData);
  gr.Mesh(*imgData);
  gr.WriteFrame("vecField.png");
  
  delete xData;
  delete yData;
  delete imgData;
}*/

#pragma mark Helpers

cv::Point unscalePoint(cv::Point p, cv::Rect origSize) {
  float ratio = (((float)kFastEyeWidth)/origSize.width);
  int x = round(p.x / ratio);
  int y = round(p.y / ratio);
  return cv::Point(x,y);
}

void scaleToFastSize(const cv::Mat &src,cv::Mat &dst) {
  cv::resize(src, dst, cv::Size(kFastEyeWidth,(((float)kFastEyeWidth)/src.cols) * src.rows));
}

cv::Mat computeMatXGradient(const cv::Mat &mat) {
  cv::Mat out(mat.rows,mat.cols,CV_64F);
  
  for (int y = 0; y < mat.rows; ++y) {
    const uchar *Mr = mat.ptr<uchar>(y);
    double *Or = out.ptr<double>(y);
    
    Or[0] = Mr[1] - Mr[0];
    for (int x = 1; x < mat.cols - 1; ++x) {
      Or[x] = (Mr[x+1] - Mr[x-1])/2.0;
    }
    Or[mat.cols-1] = Mr[mat.cols-1] - Mr[mat.cols-2];
  }
  
  return out;
}

#pragma mark Main Algorithm

void testPossibleCentersFormula(int x, int y, const cv::Mat &weight,double gx, double gy, cv::Mat &out) {
  // for all possible centers
  for (int cy = 0; cy < out.rows; ++cy) {
    double *Or = out.ptr<double>(cy);
    const unsigned char *Wr = weight.ptr<unsigned char>(cy);
    for (int cx = 0; cx < out.cols; ++cx) {
      if (x == cx && y == cy) {
        continue;
      }
      // create a vector from the possible center to the gradient origin
      double dx = x - cx;
      double dy = y - cy;
      // normalize d
      double magnitude = sqrt((dx * dx) + (dy * dy));
      dx = dx / magnitude;
      dy = dy / magnitude;
      double dotProduct = dx*gx + dy*gy;
      dotProduct = std::max(0.0,dotProduct);
      // square and multiply by the weight
      if (kEnableWeight) {
        Or[cx] += dotProduct * dotProduct * (Wr[cx]/kWeightDivisor);
      } else {
        Or[cx] += dotProduct * dotProduct;
      }
    }
  }
}

cv::Point findEyeCenter(cv::Mat face, cv::Rect eye, std::string debugWindow) {
  //face is a matrix for the entire face, "eye" is just the rectangular dimensions of where the eye is located
  
  cv::Mat eyeROIUnscaled = face(eye);
  cv::Mat eyeROI;
  scaleToFastSize(eyeROIUnscaled, eyeROI);
  // draw eye region. TODO: what is 1234?
  rectangle(face,eye,1234);
  //-- Find the gradient
  cv::Mat gradientX = computeMatXGradient(eyeROI);
  cv::Mat gradientY = computeMatXGradient(eyeROI.t()).t();
  //-- Normalize and threshold the gradient
  // compute all the magnitudes
  cv::Mat mags = matrixMagnitude(gradientX, gradientY);
  //compute the threshold
    //TODO: figure out what this is
    //The algorithm removes all gradients that are below the threshold = 0.3 * stdev(Magnitude of Gradient) + Mean(Magnitude of Gradient)
  double gradientThresh = computeDynamicThreshold(mags, kGradientThreshold);
  //double gradientThresh = kGradientThreshold;
  //double gradientThresh = 0;
  //normalize
    
  for (int y = 0; y < eyeROI.rows; ++y) {
    double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
    const double *Mr = mags.ptr<double>(y);
    for (int x = 0; x < eyeROI.cols; ++x) {
      double gX = Xr[x], gY = Yr[x];
      double magnitude = Mr[x];
      if (magnitude > gradientThresh) {
        Xr[x] = gX/magnitude;
        Yr[x] = gY/magnitude;
      } else {
        Xr[x] = 0.0;
        Yr[x] = 0.0;
      }
    }
  }
    //TODO: this makes no sense. The "mags" matrix is a white image with black dots scattered about.
    //Mags matrix is just supposed to be a matrix calculating the magnitude of the gradient at the point. It's not supposed to be an image.
    //imshow(debugWindow,mags);

    
  //-- Create a blurred and inverted image for weighting
    
    //this is an image where the iris is white, and sclera is black
    //there's also a gaussian blur on it which means it's lower resolution. TODO: find out why there's a gaussian blur
    //Gaussian blur helps to reduce image noise
  cv::Mat weight;
  GaussianBlur( eyeROI, weight, cv::Size( kWeightBlurSize, kWeightBlurSize ), 0, 0 );
  for (int y = 0; y < weight.rows; ++y) {
    unsigned char *row = weight.ptr<unsigned char>(y);
    for (int x = 0; x < weight.cols; ++x) {
      row[x] = (255 - row[x]);
    }
  }
    
  //imshow(debugWindow,weight);
  //-- Run the algorithm!
  cv::Mat outSum = cv::Mat::zeros(eyeROI.rows,eyeROI.cols,CV_64F);
  // for each possible gradient location
  // Note: these loops are reversed from the way the paper does them
  // it evaluates every possible center for each gradient location instead of
  // every possible gradient location for every center.
  printf("Eye Size: %ix%i\n",outSum.cols,outSum.rows);
  for (int y = 0; y < weight.rows; ++y) {
    const double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
    for (int x = 0; x < weight.cols; ++x) {
      double gX = Xr[x], gY = Yr[x];
      if (gX == 0.0 && gY == 0.0) {
        continue;
      }
      testPossibleCentersFormula(x, y, weight, gX, gY, outSum);
    }
  }
  // scale all the values down, basically averaging them
  double numGradients = (weight.rows*weight.cols);
  cv::Mat out;
  outSum.convertTo(out, CV_32F,1.0/numGradients); //Convert the bit depth to C32F for use in further functions and scale all the values down
    //Depth basically refers to the amount of recision used to store the pixels in an image
  //imshow(debugWindow,out);
  //-- Find the maximum point
  cv::Point maxP;
  double maxVal;
  cv::minMaxLoc(out, NULL,&maxVal,NULL,&maxP); //Calculate max and min values of the matrix and store it in the given variable
  //-- Flood fill the edges
  if(kEnablePostProcess) {
    cv::Mat floodClone;
    //double floodThresh = computeDynamicThreshold(out, 1.5);
    double floodThresh = maxVal * kPostProcessThreshold; //Compute post-processing threshold
    cv::threshold(out, floodClone, floodThresh, 0.0f, cv::THRESH_TOZERO); //Function used to apply an adaptive threshold on the image and set points below the threshold to zero
    if(kPlotVectorField) {
      //plotVecField(gradientX, gradientY, floodClone);
      imwrite("eyeFrame.png",eyeROIUnscaled);
    }
    cv::Mat mask = floodKillEdges(floodClone); //Returns a mask of the image by removing all gradients above a particular threshold since those are likely associated with hair strands or reflection from glasses
    //imshow(debugWindow + " Mask",mask);
    //imshow(debugWindow,out);
    // redo max
    cv::minMaxLoc(out, NULL,&maxVal,NULL,&maxP,mask); //Determine the maximum point of the remaining values and the point obtained is our estimation of the center
  }
  return unscalePoint(maxP,eye); //Return back the unscaled point estimate of the center
}

#pragma mark Postprocessing

bool floodShouldPushPoint(const cv::Point &np, const cv::Mat &mat) {
  return inMat(np, mat.rows, mat.cols); //Check if the point lies within the matrix
}

// returns a mask
cv::Mat floodKillEdges(cv::Mat &mat) {
    //Post-processing method to remove all the remaining values that are connected to the border of the image
  rectangle(mat,cv::Rect(0,0,mat.cols,mat.rows),255);
  
  cv::Mat mask(mat.rows, mat.cols, CV_8U, 255);
  std::queue<cv::Point> toDo;
  toDo.push(cv::Point(0,0));
  while (!toDo.empty()) {
    cv::Point p = toDo.front();
    toDo.pop();
    if (mat.at<float>(p) == 0.0f) {
        //Remove all the values that are not zero and are connected to one of the image borders
      continue;
    }
    // add in every direction to go through all points connected to the image borders
    cv::Point np(p.x + 1, p.y); // right
    if (floodShouldPushPoint(np, mat)) toDo.push(np);
    np.x = p.x - 1; np.y = p.y; // left
    if (floodShouldPushPoint(np, mat)) toDo.push(np);
    np.x = p.x; np.y = p.y + 1; // down
    if (floodShouldPushPoint(np, mat)) toDo.push(np);
    np.x = p.x; np.y = p.y - 1; // up
    if (floodShouldPushPoint(np, mat)) toDo.push(np);
    // kill it
    mat.at<float>(p) = 0.0f;
    mask.at<uchar>(p) = 0;
  }
  return mask;
}

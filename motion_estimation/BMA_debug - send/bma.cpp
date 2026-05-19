#include "wl.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// frame index
int soframe;

// mv[frame index][macroblock index][component]
int mv[20][396][2];

// Function to estimate motion vector for a macroblock located at (mx,my)
void MotionEstimate(PictImage &Anchor, // Reference frame, the previous frame
                    PictImage &Target, // Current frame that needs to be compressed
                    int mx, int my,    // Macroblock position in x and y direction
                    int &vx, int &vy)  // Output motion vector in x and y direction
{

  // Calculate the starting pixel position of the macroblock
  // The left most pixel in a single macroblock
  int start_x = mx*16; 
  int start_y = my*16;

  // Coordinate of a pixel inside a macroblock
  int dx, dy;

  int vec_x, vec_y;     // Displacement vector in x and y direction
  int DFD;              // Difference of Frame Difference
  int min_DFD = 2<<29;  // 2^30

  // Motion vector search range [-7, 7] x [-7, 7]
  for(vec_x=-7; vec_x<=7; vec_x++){
    for(vec_y=-7; vec_y<=7; vec_y++)
    {
      // DFD computation by comparing pixel by pixel in a single macroblock
      DFD = 0;
      for(dy=0; dy<16; dy++){ 
        for(dx=0; dx<16; dx++){ 

          // Formula to calculate displacement
          DFD += abs    (Anchor.Y.Get_pixel(start_x+dx, start_y+dy)               // The reference frame    
                      - Target.Y.Get_pixel(start_x+dx+vec_x, start_y+dy+vec_y));  // Target frame after + the displacement
        }
      }
      
      // Choose the best motion vector
      if(DFD < min_DFD)
    {   
        // Assign value to the target vector
        vx = vec_x;
        vy = vec_y;

        // Change min_DFD to current DFD for the next estimation
        min_DFD = DFD;

        // Store the motion vector
        mv[soframe][my*22+mx][0]=vx;
        mv[soframe][my*22+mx][1]=vy;
      }
    }
  }
}


void MotionCompensate(PictImage &Target, 
                      PictImage &Predicted,
                      int mx, int my, 
                      int vx, int vy) 
// lay duoc MV so sanh voi Prev Frame, dong thoi MV nay cung  
// se duoc su dung ngay de bu dap cho MB trong curr_frame (khong can phai doi dau gi ca)..=> goi la QT compensate
{

  // Determine the absolute position of current pixel
  int start_x = mx * 16;
  int start_y = my * 16;

  int dx, dy;

  // Motion compensating for Y
  for(dx=0; dx<16; dx++){
    for(dy=0; dy<16; dy++)
      {
        // Ghi giá trị pixel vào frame dự đoán
        Predicted.Y.Get_pixel(start_x+dx, start_y+dy) 
        = Target.Y.Get_pixel(start_x+dx+vx, start_y+dy+vy); // Lấy pixel từ frame tham chiếu (target) tại vị trí dịch chuyển bởi motion vector (vx,vy)
      }
  }

  // divide the motion vector by 2 to compensate U and V component
  vx = vx/2;
  vy = vy/2;
  
  start_x = start_x/2;
  start_y = start_y/2;

  // Motion compensating for U and V
  // Duyệt qua từng pixel của U và V
  for(dx=0; dx<8; dx++){
    for(dy=0; dy<8; dy++)
      { 
        // Copy block U từ frame tham chiếu
        Predicted.U.Get_pixel(start_x+dx, start_y+dy)
          = Target.U.Get_pixel(start_x+dx+vx, start_y+dy+vy);
        
        // Giống U
        Predicted.V.Get_pixel(start_x+dx, start_y+dy)
          = Target.V.Get_pixel(start_x+dx+vx, start_y+dy+vy);
      }
  }
}

// Function to perform motion estimation for all macroblocks and then get the predicted image
void MotionEstimateAndGetPredictedImage(
                                        PictImage &Anchor,      
                                        PictImage &Target, 
                                        PictImage &Predicted)
{

  // Get the Image dimensions
  int ImageWidth = Anchor.Y.GetWidthOfImage();
  int ImageHeight = Anchor.Y.GetHeightOfImage();

  // Number of MBs calculated by dimensions of the Image
  int MX = ImageWidth/16;  // # of Macroblocks (MBs) in x direction
  int MY = ImageHeight/16; // # of Macroblocks in y direction
  
  // MBs index 
  int mx, my;
  
  int vx, vy; // motion vector

  // Iterate over all the MBs
  for(mx=0; mx<MX; mx++) 
	  for(my=0; my<MY; my++) // for each MB
  {
    // estimate its motion vector
    MotionEstimate(Anchor, Target, mx, my, vx, vy); 
  
    // Debug
	for (int kk=0;kk<10;kk++)
		printf("Frame: %d, MB[%d], Mx=%d, My=%d \n",soframe,kk,mv[soframe][kk][0],mv[soframe][kk][1]);

    // copy and paste the matched block (specified by the motion vector)
    // into the Predicted image
    MotionCompensate(Target, Predicted, mx, my, vx, vy); 
  }
}


void main()
{

  // Initializations
  PictImage Target(352,288); 
  PictImage Anchor(352,288); 
  PictImage Predicted(352,288); 

  // from foreman sequence, read the 0th frame whose Y size is 352x288 
  Anchor.Load("FOREMAN_CIF15.yuv", 352, 288, 0); 

  // because there is no predicted frame for the 0th frame, just save 
  // it to "predicted.qcif"
  Anchor.Save("predicted.qcif");

  // for the time measurement
  clock_t start, finish;
  double  duration; 

  start = clock();
  
  // do the motion estimation for the next 99 frames
  for(int frame_no=1; frame_no<100; frame_no++)
  {
  soframe=frame_no;
	Target = Anchor; // Frame trước trở thành tham chiếu mới
    Anchor.Load("FOREMAN_CIF15.yuv", 352, 288, frame_no); // Đọc frame hiện tại

    MotionEstimateAndGetPredictedImage(Anchor, Target, Predicted); // Tạo frame dự đoán

    fprintf(stdout, "PSNR of %dth frame = %f\n", frame_no, psnr(Anchor.Y, Predicted.Y)); // Tính PSNR

    Predicted.Save("predicted.qcif", 1); // last arg = 1 means that appending to the file 
    // File này chứa chuỗi frame dự đoán từ 0 đến 99
  }

  // Kết thúc đo thời gian, in ra
  finish = clock();
  duration = (double)(finish - start) / CLOCKS_PER_SEC;
  fprintf(stdout, "It took %2.1f seconds\n", duration );
}







	// // tamm thoi thoi:
	// nMinPosX = nSadSize/2;
	// nMinPosY = nSadSize/2;
	// min_DFD=65000;
	// for( nY=0;nY<nSadSize;nY++)
	// for( nX=0;nX<nSadSize;nX++)
	// {
	// 	DFD = 0;
	// 	for( nY1 = 0 ; nY1 < nBlockSize ; nY1++){    
	// 	for( nX1 = 0 ; nX1 < nBlockSize ; nX1++){ 
	// 		nTempMerged_nxt[nY1*nBlockSize+nX1]=nNxtTemp2D[offset+bi*nBlockSize+nY1]
	// 		[offset+bj*nBlockSize+nX1];     // MacroBlock 1
	// 		nTempMerged_nxt[nBlockSize*nBlockSize+nY1*nBlockSize+nX1]=nNxtTemp2D[offset+bi*nBlockSize+nY1]
	// 		[offset+(bj+1)*nBlockSize+nX1]; // MacroBlock 2

	// 		nTempMerged_prv[nY1*nBlockSize+nX1]=nPreTemp2D[offset+bi*nBlockSize+nY1+nY-nSadSize/2]
	// 		[offset+bj*nBlockSize+nX1+nX-nSadSize/2];     // MacroBlock 1
	// 		nTempMerged_prv[nBlockSize*nBlockSize+nY1*nBlockSize+nX1]=nPreTemp2D[offset+bi*nBlockSize+nY1+nY-nSadSize/2]
	// 		[offset+(bj+1)*nBlockSize+nX1+nX-nSadSize/2]; // MacroBlock 2
										
	// 		// tinh DFD:

	// 		DFD+= abs(nTempMerged_nxt[nY1*nBlockSize+nX1]-nTempMerged_prv[nY1*nBlockSize+nX1]) 
	// 			 +abs(nTempMerged_nxt[nBlockSize*nBlockSize+nY1*nBlockSize+nX1]-nTempMerged_prv[nBlockSize*nBlockSize+nY1*nBlockSize+nX1]);
	// 	}} 
		
	// 	if(nX==nSadSize/2 && nY==nSadSize/2)
	// 	{
	// 		if(DFD<min_DFD+(delta))
	// 		{
	// 			nMinPosX = nX;
	// 			nMinPosY = nY;
	// 			min_DFD  = (DFD-(delta));
	// 		}
	// 	}
	// 	else
	// 	{
	// 		if(DFD<min_DFD)
	// 		{
	// 			nMinPosX = nX;  
	// 			nMinPosY = nY;
	// 			min_DFD  = DFD;
	// 		}				
	// 	}
	// 	MVy[bi][bj]=(nMinPosY-nSadSize/2);
	// 	MVx[bi][bj]=(nMinPosX-nSadSize/2);
	// 	MVy[bi][bj+1]=(nMinPosY-nSadSize/2);
	// 	MVx[bi][bj+1]=(nMinPosX-nSadSize/2);
	// }
	// // sau khi ket thuc tim ra duoc opt_MV => dong thoi thuc hien luon viec Re-classify 
	// // trong vong if nay doi voi nhom cac MBs nay luon=> tach thanh cac block 8x8
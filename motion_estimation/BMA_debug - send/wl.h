#ifndef _WL_H_
#define _WL_H_

// unsigned char is 8 bits data type
// define BYTE as an alias of unsigned char
#define BYTE unsigned char 

// Forward declaration to avoid circular include
class CharImage;
class PictImage;

// Base class for Image Library
struct IL{
void Error(char* msg);
void Warning(char* msg);
};

// Class to represent a gray scale image 
// Inhertit member functions from IL
class CharImage:public IL{
public:
	BYTE* pBuffer; // pointer to the image buffer (1D array), 
	// it points to the area of memory that has the type of BYTE which is unsigned char = 8 bits
	// it means we use 8bits to represent a pixel value (0-255)

	// image width and height
	int WidthOfImage;
	int HeightOfImage;
		
	// CONSTRUCTOR PROTOTYPES
	// Create a blank image (an 2D array) with given width and height
	CharImage(int width = 352, int height = 288);

	// Create a 2D array with dimension width x height and load image from a file
	CharImage(char* FileName, int width = 352, int height = 288);

	// reference function to overload the assignment operator
	// This function is used to copy an image to another image
	// Overwrite the existing image buffer with the new image buffer
	CharImage& operator=(CharImage& CI);
	
	// Destructor prototype 
	~CharImage();
	
	// Function to access pixel value at (X,Y)
	// Can be used to adjust or get pixel value
	BYTE& Get_pixel(int X,int Y){

		// If X < Xmin, clip it to Xmin = 0
		if(X<0) X=0;

		// If X > Xmax, clip it to Xmax = WidthOfImage-1
		else if(X>WidthOfImage-1) X=WidthOfImage-1;

		// If Y < Ymin, clip it to Ymin = 0
		if(Y<0) Y=0;

		// If Y > Ymax, clip it to Ymax = HeightOfImage-1
		else if(Y>HeightOfImage-1) Y = HeightOfImage-1;
		
		// Return the reference to the pixel (X,Y)
		// This is how we get the pixel (X,Y) - 2D array with pointer that point to a 1D array
		// Offset = Y*WidthOfImage + X
		return *(pBuffer+(Y*WidthOfImage+X));
	}
	
	// Get functions to return width, height and buffer pointer
	int GetWidthOfImage(){ return WidthOfImage; }
	int GetHeightOfImage(){ return HeightOfImage; }
	BYTE* GetBufferw(){ return pBuffer; }
	
	// Function to clean up the image buffer
	void clean();

	// Load image from a file but with a new buffer 
	// # Constructor already allocates memory for pBuffer
	void Load(char* FileName, int width = 352, int height = 288, int offset=0);

	// Save image to a file
	void Save(char* FileName, int append=0);
};


// This class represents a YUV picture - which has Y - Lumninance, U - Chrominance, V - Chrominance components
class PictImage :public CharImage
{

	// Inhertit member functions from CharImage and IL
public: 
  CharImage Y; // Luminance - gray scale

  // These two components contain color infomation
  CharImage U;
  CharImage V;
  
  // CONSTRUCTOR PROTOTYPES
  PictImage(char *filename, int width=352, int height=288);
  PictImage(int width = 352, int height=288);
  PictImage& operator=(PictImage& CI);


  // Function to clean up the image buffer
  void clean()
  {
    Y.clean();
    U.clean();
    V.clean();
  };



  // These two functions are the same role as in CharImage class
  void Load(char *fileName, int width = 352, int height = 288, int frame_no=0);
  void Save(char *filename, int append = 0);

};

// Function to compute PSNR between two gray scale images
float psnr(CharImage& A, CharImage& B);

#endif

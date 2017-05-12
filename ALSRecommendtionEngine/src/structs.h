#ifndef STRUCTS_H
#define STRUCTS_H


struct Info{
	// numRatings -> how many movies has this user rated/By how many users htis movie been rated
	// rating     -> array containing the ratings os this user/movie
	// rId        -> array containing the coresponding id of the movie/user 
	// id         -> id of the user/movie
	// if for example rating[0] = 5 && rId[0] = 6 this means either the user rated movie 6 with 5 either 
	// this movie has been rated with 5 by user 6
	int numRatings;
	float* rating;
	int* rId; 
	int id;
};
typedef struct Info info;


struct sparse_entry{
	// a convenient struct to read the data from the file
	// we use this struct to structre the data for the Info struct
	int rowUser;
	int colMovie;
	float rating;
};
typedef struct sparse_entry sparseEntry;


#endif

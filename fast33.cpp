cv::Mat img_object=cv::imread("templ.bmp", 0);
std::vector<cv::KeyPoint> keypoints_object, keypoints_scene;
cv::Mat descriptors_object, descriptors_scene;

cv::ORB orb(500, 1.2, 4, 31, 0, 2, 0, 31);

// особые точки объекта
orb.detect(img_object, keypoints_object);
orb.compute(img_object, keypoints_object, descriptors_object);

// особые точки картинки
cv::Mat img = cv::imread("img.bmp", 1);
cv::Mat img_scene = cv::Mat(img.size(), CV_8UC1);
orb.detect(img, keypoints_scene);
orb.compute(img, keypoints_scene, descriptors_scene);

cv::imshow("desrs", descriptors_scene);
cvWaitKey();
int test[10];
for (int q = 0; q<10 ; q++) test[q]=descriptors_scene.data[q];

//-- matching descriptor vectors using FLANN matcher
cv::BFMatcher matcher;
std::vector<cv::DMatch> matches;
cv::Mat img_matches;
if(!descriptors_object.empty() && !descriptors_scene.empty()) {
    matcher.match (descriptors_object, descriptors_scene, matches);
    double max_dist = 0; double min_dist = 100;

    // calculation of max and min idstance between keypoints
    for( int i = 0; i < descriptors_object.rows; i++)
    { double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }

    //-- Draw only good matches (i.e. whose distance is less than 3*min_dist)
    std::vector< cv::DMatch >good_matches;
    for( int i = 0; i < descriptors_object.rows; i++ ) if( matches[i].distance < (max_dist/1.6) ) good_matches.push_back(matches[i]); 
    
    cv::drawMatches(img_object, keypoints_object, img_scene, keypoints_scene, good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), 
	std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
}

cv::imshow("match result", img_matches );
cv::waitKey();

return 0;

package com.example.deg032.opencvwithsift;

import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Spinner;
import android.widget.Toast;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.xfeatures2d.SIFT;


public class MainActivity extends AppCompatActivity implements AdapterView.OnItemSelectedListener {

    private static final String TAG = "MainActivity";
    //private Mat mImage = new Mat();

    public static Double nResolutionDivider = 8.0;

    public static Integer MIN_MATCH_COUNT = 300;

    public static Mat objImageMat;

    public static MatOfKeyPoint mRefKeyPoints;

    public static Mat mRefDescriptors;

    private Camera2BasicFragment mCameraFragment;

    static {
        if(!OpenCVLoader.initDebug()){
            Log.d(TAG, "OpenCV not loaded");
        } else {
            Log.d(TAG, "OpenCV loaded");
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        //mDisplayText = (TextView) findViewById(R.id.display_text);

        Spinner spinner = (Spinner) findViewById(R.id.image_size_spinner);
        spinner.setOnItemSelectedListener(this);

        // Create an ArrayAdapter using the string array and a default spinner layout
        ArrayAdapter<CharSequence> adapter = ArrayAdapter.createFromResource(this,
                R.array.image_size_array, android.R.layout.simple_spinner_item);
        // Specify the layout to use when the list of choices appears
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        // Apply the adapter to the spinner
        spinner.setAdapter(adapter);

        RefImageMat();

        //mNetworkFragment = NetworkFragment.getInstance(getSupportFragmentManager(), "https://www.google.com");

        //testOpenCVSift();

        if (null == savedInstanceState) {
            mCameraFragment = Camera2BasicFragment.newInstance();
            getSupportFragmentManager().beginTransaction()
                    .replace(R.id.image_container, mCameraFragment)
                    .commit();
        }
    }

    @Override
    public void onItemSelected(AdapterView<?> parent, View view,
                               int pos, long id) {
        String sResolution = parent.getItemAtPosition(pos).toString();
        switch(sResolution){
            case "12":
                nResolutionDivider = 1.0;
                break;
            case "8.5":
                nResolutionDivider = 1.2;
                break;
            case "4.8":
                nResolutionDivider = 1.6;
                break;
            case "3":
                nResolutionDivider = 2.0;
                break;
            case "2.1":
                nResolutionDivider = 2.4;
                break;
            case "1.5":
                nResolutionDivider = 2.8;
                break;
            case "1.2":
                nResolutionDivider = 3.2;
                break;
            case "1":
                nResolutionDivider = 3.6;
                break;
            case "0.8":
                nResolutionDivider = 4.0;
                break;
            case "0.5":
                nResolutionDivider = 5.0;
                break;
            case "0.3":
                nResolutionDivider = 6.0;
                break;
            case "0.25":
                nResolutionDivider = 7.0;
                break;
            case "0.2":
                nResolutionDivider = 8.0;
                break;
            default:
                nResolutionDivider = 2.4;
                break;
        }
        // Showing selected spinner item
        Toast.makeText(parent.getContext(), "Selected OpenCV image resolution: " + sResolution, Toast.LENGTH_LONG).show();
    }

    public void onNothingSelected(AdapterView<?> parent) {
        nResolutionDivider = 2.4;
    }

    /** Just for testing SIFT. */
    private void RefImageMat(){
        mRefKeyPoints = new MatOfKeyPoint();
        mRefDescriptors = new Mat();
        objImageMat = new Mat();

        try {
            objImageMat = Utils.loadResource(MainActivity.this, R.drawable.train, Imgcodecs.CV_LOAD_IMAGE_COLOR);
            SIFT mFeatureDetector = SIFT.create();

            Log.d(TAG, "Height: " + Integer.toString(objImageMat.height())
                    + ", Width: " + Integer.toString(objImageMat.width()));

            long time = System.currentTimeMillis();

            mFeatureDetector.detect(objImageMat, mRefKeyPoints);
            mFeatureDetector.compute(objImageMat, mRefKeyPoints, mRefDescriptors);
            Log.d(TAG, "Time to process " + (System.currentTimeMillis() - time) +
                    ", Number of key points: " + mRefKeyPoints.toArray().length);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}
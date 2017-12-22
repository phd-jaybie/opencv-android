package com.example.deg032.opencvwithsift;

import android.app.Activity;
import android.app.AlertDialog;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.EditText;
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

import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;

import javax.net.ssl.HttpsURLConnection;


public class MainActivity extends AppCompatActivity implements AdapterView.OnItemSelectedListener {

    private static final String TAG = "MainActivity";

    private static final Integer INAPP = 1;

    private static final Integer EDGE = 2;

    private static final Integer CLOUD = 3;

    private static final Integer STORE = 4;

    public static Integer operatingMode = 1;
    //private Mat mImage = new Mat();

    private HandlerThread backgroundThread;

    private Handler backgroundHandler;

    public static Double nResolutionDivider = 8.0;

    public static Integer MIN_MATCH_COUNT = 40;

    public static Mat objImageMat;

    public static MatOfKeyPoint mRefKeyPoints;

    public static Mat mRefDescriptors;

    public static String edgeURL = "192.168.43.98";

    private Camera2BasicFragment mCameraFragment;

    private EditText edgeIP;

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
        startBackgroundThread();

        Spinner resolutionSpinner = findViewById(R.id.image_size_spinner);
        resolutionSpinner.setOnItemSelectedListener(this);

        Spinner modeSpinner = findViewById(R.id.mode_spinner);
        modeSpinner.setOnItemSelectedListener(this);

        edgeIP = findViewById(R.id.internet_address_Text);

        // Create an ArrayAdapter using the string array and a default spinner layout
        ArrayAdapter<CharSequence> resolutionAdapter = ArrayAdapter.createFromResource(this,
                R.array.image_size_array, android.R.layout.simple_spinner_item);
        resolutionAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        resolutionSpinner.setAdapter(resolutionAdapter);

        ArrayAdapter<CharSequence> modeAdapter = ArrayAdapter.createFromResource(this,
                R.array.mode_array, android.R.layout.simple_spinner_item);
        modeAdapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        modeSpinner.setAdapter(modeAdapter);

        /** Extract the reference SIFT features */
        backgroundHandler.post(new RefImageMat());

        if (null == savedInstanceState) {
            mCameraFragment = Camera2BasicFragment.newInstance();
            getSupportFragmentManager().beginTransaction()
                    .replace(R.id.image_container, mCameraFragment)
                    .commit();
        }
    }

    private void imageResolutionSelected(AdapterView<?> parent, int pos) {
        String sResolution = parent.getItemAtPosition(pos).toString();
        switch(sResolution){
            case "12.2":
                nResolutionDivider = 1.0;
                MIN_MATCH_COUNT = 300;
                break;
            case "10.1":
                nResolutionDivider = 1.1;
                MIN_MATCH_COUNT = 280;
                break;
            case "8.5":
                nResolutionDivider = 1.2;
                MIN_MATCH_COUNT = 250;
                break;
            case "4.8":
                nResolutionDivider = 1.6;
                MIN_MATCH_COUNT = 190;
                break;
            case "3":
                nResolutionDivider = 2.0;
                MIN_MATCH_COUNT = 150;
                break;
            case "2.1":
                nResolutionDivider = 2.4;
                MIN_MATCH_COUNT = 130;
                break;
            case "1.5":
                nResolutionDivider = 2.8;
                MIN_MATCH_COUNT = 110;
                break;
            case "1.2":
                nResolutionDivider = 3.2;
                MIN_MATCH_COUNT = 100;
                break;
            case "1":
                nResolutionDivider = 3.6;
                MIN_MATCH_COUNT = 90;
                break;
            case "0.8":
                nResolutionDivider = 4.0;
                MIN_MATCH_COUNT = 80;
                break;
            case "0.5":
                nResolutionDivider = 5.0;
                MIN_MATCH_COUNT = 60;
                break;
            case "0.3":
                nResolutionDivider = 6.0;
                MIN_MATCH_COUNT = 50;
                break;
            case "0.25":
                nResolutionDivider = 7.0;
                MIN_MATCH_COUNT = 50;
                break;
            case "0.2":
                nResolutionDivider = 8.0;
                MIN_MATCH_COUNT = 40;
                break;
            case "0.151":
                nResolutionDivider = 9.0;
                MIN_MATCH_COUNT = 30;
                break;
            case "0.122":
                nResolutionDivider = 10.0;
                MIN_MATCH_COUNT = 30;
                break;
            case "0.101":
                nResolutionDivider = 11.0;
                MIN_MATCH_COUNT = 30;
                break;
            case "0.085":
                nResolutionDivider = 12.0;
                MIN_MATCH_COUNT = 30;
                break;
            case "0.072":
                nResolutionDivider = 13.0;
                MIN_MATCH_COUNT = 30;
                break;
            case "0.062":
                nResolutionDivider = 14.0;
                MIN_MATCH_COUNT = 30;
                break;
            case "0.054":
                nResolutionDivider = 15.0;
                MIN_MATCH_COUNT = 20;
                break;
            case "0.042":
                nResolutionDivider = 17.0;
                MIN_MATCH_COUNT = 20;
                break;
            case "0.03":
                nResolutionDivider = 20.0;
                MIN_MATCH_COUNT = 20;
                break;
            case "0.021":
                nResolutionDivider = 24.0;
                MIN_MATCH_COUNT = 20;
                break;
            case "0.014":
                nResolutionDivider = 29.0;
                MIN_MATCH_COUNT = 20;
                break;
            case "0.009":
                nResolutionDivider = 36.0;
                MIN_MATCH_COUNT = 10;
                break;
            case "0.007":
                nResolutionDivider = 43.0;
                MIN_MATCH_COUNT = 10;
                break;
            case "0.005":
                nResolutionDivider = 51.0;
                MIN_MATCH_COUNT = 10;
                break;
            case "0.003":
                nResolutionDivider = 60.0;
                MIN_MATCH_COUNT = 10;
                break;
            case "0.0025":
                nResolutionDivider = 70.0;
                MIN_MATCH_COUNT = 10;
                break;
            case "0.0019":
                nResolutionDivider = 81.0;
                MIN_MATCH_COUNT = 10;
                break;
            case "0.0014":
                nResolutionDivider = 93.0;
                MIN_MATCH_COUNT = 10;
                break;
            case "0.0011":
                nResolutionDivider = 106.0;
                MIN_MATCH_COUNT = 10;
                break;
            case "0.0008":
                nResolutionDivider = 122.0;
                MIN_MATCH_COUNT = 10;
                break;
            case "0.0006":
                nResolutionDivider = 137.0;
                MIN_MATCH_COUNT = 10;
                break;
            default:
                nResolutionDivider = 8.0;
                MIN_MATCH_COUNT = 40;
                break;
        }
        // Showing selected spinner item
        Toast.makeText(parent.getContext(), "Selected OpenCV image resolution: " + sResolution, Toast.LENGTH_LONG).show();
    }

    private void modeSelected(AdapterView<?> parent, int pos) {
        String sMode = parent.getItemAtPosition(pos).toString();
        switch(sMode){
            case "In-app":
                operatingMode = INAPP;
                break;
            case "Edge":
                edgeURL = edgeIP.getText().toString();
                operatingMode = EDGE;
                break;
            case "Cloud":
                operatingMode = CLOUD;
                break;
            case "Store":
                operatingMode = STORE;
                break;
            default:
                operatingMode = INAPP;
                break;
        }
        // Showing selected spinner item
        Toast.makeText(parent.getContext(), "Selected Mode: " + sMode, Toast.LENGTH_LONG).show();
    }

    @Override
    public void onItemSelected(AdapterView<?> parent, View view,
                               int pos, long id) {
        Spinner spinner = (Spinner) parent;
        switch (spinner.getId()) {
            case R.id.image_size_spinner: {
                imageResolutionSelected(parent, pos);
                break;
            }
            case R.id.mode_spinner: {
                modeSelected(parent, pos);
                break;
            }
        }
    }

    public void onNothingSelected(AdapterView<?> parent) {
        MIN_MATCH_COUNT = 40;
        nResolutionDivider = 8.0;
        operatingMode = INAPP;
    }

    /** Just for testing SIFT. */
    private class RefImageMat implements Runnable {

        @Override
        public void run() {

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

    private void startBackgroundThread() {
        backgroundThread = new HandlerThread("MainBackground");
        backgroundThread.start();
        backgroundHandler = new Handler(backgroundThread.getLooper());
    }

    /**
     * Stops the background thread and its {@link Handler}.
     */
    private void stopBackgroundThread() {
        backgroundThread.quitSafely();
        try {
            backgroundThread.join();
            backgroundThread = null;
            backgroundHandler = null;
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    @Override
    public void onResume() {
        super.onResume();
        startBackgroundThread();
    }

    @Override
    public void onPause() {
        stopBackgroundThread();
        super.onPause();
    }


}
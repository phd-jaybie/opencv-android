package com.example.deg032.opencvwithsift;

import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;

import org.opencv.android.OpenCVLoader;


public class MainActivity extends AppCompatActivity {

    private static final String TAG = "MainActivity";
    //private Mat mImage = new Mat();

    //private TextView mDisplayText;

    private Camera2BasicFragment mCameraFragment;
    private NetworkFragment mNetworkFragment;


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
        mNetworkFragment = NetworkFragment.getInstance(getSupportFragmentManager(), "https://www.google.com");

        if (null == savedInstanceState) {
            mCameraFragment = Camera2BasicFragment.newInstance();
            getSupportFragmentManager().beginTransaction()
                    .replace(R.id.image_container, mCameraFragment)
                    .commit();
        }
    }

}
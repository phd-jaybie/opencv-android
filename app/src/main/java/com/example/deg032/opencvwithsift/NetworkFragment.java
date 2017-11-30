package com.example.deg032.opencvwithsift;

import android.os.Bundle;
import android.support.v4.app.Fragment;
import android.support.v4.app.FragmentManager;

/**
 * Created by deg032 on 29/11/17.
 */

public class NetworkFragment extends Fragment {

    public static final String TAG = "NetworkFragment";

    private static final String URL_KEY = "UrlKey";
    /**
     * Static initializer for NetworkFragment that sets the URL of the host it will be downloading
     * from.
     */
    public static NetworkFragment getInstance(FragmentManager fragmentManager, String url) {
        NetworkFragment networkFragment = new NetworkFragment();
        Bundle args = new Bundle();
        args.putString(URL_KEY, url);
        networkFragment.setArguments(args);
        fragmentManager.beginTransaction().add(networkFragment, TAG).commit();
        return networkFragment;
    }

}

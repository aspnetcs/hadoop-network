package com.tencent.bi.utils;

import java.net.InetAddress;

import cern.colt.matrix.DoubleMatrix2D;

public class StringUtils {

	public static String array2String(double[] array) {
		StringBuilder res = new StringBuilder("");
		for (int i = 0; i < array.length; i++) {
			if (i != 0)
				res.append(",");
			res.append(array[i]);
		}
		return res.toString();
	}

	public static String matrix2String(DoubleMatrix2D M) {
		StringBuilder res = new StringBuilder("");
		for (int i = 0; i < M.rows(); i++)
			for (int j = 0; j < M.columns(); j++) {
				if (i != 0 || j != 0)
					res.append(",");
				res.append(M.getQuick(i, j));
			}
		return res.toString();
	}

	public static String getIP() {
		try {
			InetAddress addr = InetAddress.getLocalHost();
			return addr.getHostAddress().toString();
		} catch (Exception e) {
			e.printStackTrace();
			return "";
		}
	}
}

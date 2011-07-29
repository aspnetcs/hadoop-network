package com.tencent.bi.utils.hadoop;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;

public class FileOperators {
	
	public static String confName = "configuration-bi.xml";
	
	public static void HDFSMove(boolean delSrc, String src, String dst) {
		FileSystem fs = null;
		Configuration conf = new Configuration();
		try {
			fs = FileSystem.get(conf);
			Path srcPath = new Path(src);
			FileStatus fsta[] = fs.globStatus(srcPath);
			for (FileStatus it : fsta) {
				Path singlePath = it.getPath();
				Path dstPath = new Path(dst);
				FileUtil.copyMerge(fs, singlePath, fs, dstPath, delSrc, conf, "");
			}
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				fs.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
	
	public static Configuration getConfiguration(){
		Configuration conf = new Configuration();
		conf.addResource(new Path(confName));
		return conf;
	}
}

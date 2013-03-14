package com.tencent.bi.utils.hadoop;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FileUtil;
import org.apache.hadoop.fs.Path;

/**
 * A utility class providing file operators on HDFs.
 * @author tigerzhong
 *
 */
public class FileOperators {
	/**
	 * Configuration file
	 */
	public static String confName = "configuration-bi.xml";
	
	/**
	 * File copy and merge on HDFS. Supports wildcards. Support directory. Not recursively copy.
	 * <p>
	 * @weixue Not a good name?
	 * @see FileUtil#copyMerge(FileSystem, Path, FileSystem, Path, boolean, Configuration, String).
	 * @param delSrc
	 * @param src
	 * @param dst
	 */
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
				//@weixue Only the first round will pass, the 2nd round will throw exceptions, since
				//the dstPath already exists.
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
	/**
	 * Get configurations from HadoopMF specific configuration file.
	 * @return
	 */
	public static Configuration getConfiguration(){
		Configuration conf = new Configuration();
		// Add another layer on top of default Hadoop configuration (default, site).
		conf.addResource(new Path(confName));
		return conf;
	}
}

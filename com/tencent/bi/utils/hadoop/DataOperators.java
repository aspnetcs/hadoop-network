package com.tencent.bi.utils.hadoop;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

/**
 * Data operators
 * @author tigerzhong
 *
 */
public class DataOperators {
	
	/**
	 * Read text from HDFS line by line
	 * @param conf, configuration
	 * @param pathName, path of data
	 * @return data in list
	 * @throws IOException
	 */
	public static List<String> readTextFromHDFS(Configuration conf, String pathName) throws IOException{
		List<String> resList = new ArrayList<String>();
		FileSystem fs = FileSystem.get(conf); 
		FileStatus fsta[] = fs.globStatus(new Path(pathName+"*"));
		for (FileStatus it : fsta) {
			Path singlePath = it.getPath();
			if(it.isDir()) continue;
			BufferedReader in = new BufferedReader(new InputStreamReader(fs.open(singlePath)));
			while(in.ready())
				resList.add(in.readLine());
			in.close();
		}
		return resList;
	}

	/**
	 * Save text to HDFS
	 * @param conf, configuration
	 * @param pathName, path of data
	 * @param data, data in list
	 * @param overWrite, whether overwrite
	 * @throws IOException
	 */
	public static void saveTextToHDFS(Configuration conf, String pathName, List<String> data, boolean overWrite) throws IOException{
		FileSystem fs = FileSystem.get(conf); 
		BufferedWriter out = new BufferedWriter(new OutputStreamWriter(fs.create(new Path(pathName),overWrite)));
		for(String line : data)
			out.write(line+"\n");
		out.close();
	}
	
	/**
	 * Append text to HDFS in line
	 * @param conf, configuration
	 * @param pathName
	 * @param data
	 * @throws IOException
	 */
	public static void appendTextToHDFS(Configuration conf, String pathName, String data) throws IOException {
		FileSystem fs = FileSystem.get(conf); 
		BufferedWriter out = new BufferedWriter(new OutputStreamWriter(fs.append(new Path(pathName))));
		out.write(data+"\n");
		out.close();
	}
	
}

package com.tencent.bi.utils.hadoop;

import java.io.IOException;


import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.DefaultStringifier;

import com.tencent.bi.cf.optimization.Loss;

/**
 * A utility class providing static methods for manipulating model specific meta data.
 * @author tigerzhong
 *
 */
public class ModelOperator {
	
	/**
	 * Save the loss object as a string
	 * @param conf
	 * @param func
	 * @return serialization string
	 * @throws IOException
	 */
	public static String loss2String(Configuration conf, Loss func) throws IOException{
		conf.set("io.serializations", "org.apache.hadoop.io.serializer.JavaSerialization,org.apache.hadoop.io.serializer.WritableSerialization");
		DefaultStringifier<Loss> output = new DefaultStringifier<Loss>(conf, Loss.class); 
		String str = output.toString(func);
		return str;
	}
	
	/**
	 * Recover a loss object from specified string
	 * @param conf
	 * @param str
	 * @return loss object
	 * @throws IOException
	 */
	public static Loss string2Loss(Configuration conf, String str) throws IOException{
		conf.set("io.serializations", "org.apache.hadoop.io.serializer.JavaSerialization,org.apache.hadoop.io.serializer.WritableSerialization");
		DefaultStringifier<Loss> ds = new DefaultStringifier<Loss>(conf, Loss.class); 
		Loss l = (Loss) ds.fromString(str);
		return l;
	}

}

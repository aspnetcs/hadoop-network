package com.tencent.bi.utils.io;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DefaultStringifier;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;

import com.tencent.bi.utils.hadoop.DataOperators;
import com.tencent.bi.utils.serialization.MatrixRowWritable;


import cern.colt.matrix.DoubleFactory2D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;

/**
 * Matrix IO and transformation operations provided as static methods of this utility class.
 * @author tigerzhong
 */
public class MatrixIO {
	
	/**
	 * Write one row of matrix to a {@link SequenceFile} in HDFS.
	 * @param name The pathname of the HDFS file to be written, which will be a {@link SequenceFile}.
	 * @param key the {@link LongWritable} key for this row.
	 * @param value the row itself.
	 */
	public static void writeMatrixRow(String name, LongWritable key, MatrixRowWritable value) throws IOException{
		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.get(conf);
		SequenceFile.Writer wr = SequenceFile.createWriter(fs, conf, new Path(name), LongWritable.class, MatrixRowWritable.class);
		wr.append(key, value);
		wr.sync();
	}
	
	/**
	 * Read dense matrix from specified text file on local file system.
	 * @param name, file name
	 * @return a newly created dense matrix as a {@link DenseDoubleMatrix2D}.
	 * @throws IOException
	 */
	public static DenseDoubleMatrix2D readDenseMatrixFromText(String name) throws IOException{
		BufferedReader fis = new BufferedReader(new FileReader(name));
		List<double[]> tmpM = new ArrayList<double[]>();
		int d = 0;
		String line = null;
		while((line=fis.readLine())!=null){
			String items[] = line.split(",",-1);
			double[] vec = new double[items.length];
			for(int i=0;i<items.length;i++)
				vec[i] = Double.parseDouble(items[i]);
			d = items.length;
			tmpM.add(vec);
		}
		fis.close();
		DenseDoubleMatrix2D M = (DenseDoubleMatrix2D) DoubleFactory2D.dense.make(tmpM.size(), d);
		for(int i=0;i<tmpM.size();i++){
			double[] item = tmpM.get(i);
			for(int j=0;j<d;j++)
				M.setQuick(i, j, item[j]);
		}
		return M;
	}
	
	/**
	 * Read dense matrix from specified text file on local file system.
	 * @param name file name
	 * @return a newly created dense matrix as a double[][] array.
	 * @throws IOException
	 */
	public static double[][] readDense2DArrayFromText(String name) throws IOException{
		BufferedReader fis = new BufferedReader(new FileReader(name));
		List<double[]> tmpM = new ArrayList<double[]>();
		int d = 0;
		String line = null;
		while((line=fis.readLine())!=null){
			String items[] = line.split(",",-1);
			double[] vec = new double[items.length];
			for(int i=0;i<items.length;i++){
				vec[i] = Double.parseDouble(items[i]);
			}
			d = items.length;
			tmpM.add(vec);
		}
		fis.close();
		double[][] M = new double[tmpM.size()][d];
		for(int i=0;i<tmpM.size();i++){
			double[] item = tmpM.get(i);
			for(int j=0;j<d;j++)
				M[i][j] = item[j];
		}
		return M;
	}
	
	/**
	 * Read a dense vector (1D array) from a file on local file system.
	 * @param name file name.
	 * @return a newly created double[] array.
	 * @throws IOException
	 */
	public static double[] readDense1DArrayFromText(String name) throws IOException{
		BufferedReader fis = new BufferedReader(new FileReader(name));
		List<Double> tmpM = new ArrayList<Double>();
		String line = null;
		while((line=fis.readLine())!=null){
			tmpM.add(Double.parseDouble(line));
		}
		fis.close();
		double[] M = new double[tmpM.size()];
		for(int i=0;i<tmpM.size();i++){
			M[i] = tmpM.get(i);
		}
		return M;
	}
	
	/**
	 * Save specified dense matrix to specified file on local file system.
	 * @param name file name.
	 * @param M the dense matrix to be written out.
	 * @throws IOException
	 */
	public static void saveDenseMatrixToText(String name, DenseDoubleMatrix2D M) throws IOException{
		BufferedWriter fw = new BufferedWriter(new FileWriter(name));
		for(int i=0;i<M.rows();i++){
			StringBuilder line = new StringBuilder("");
			for(int j=0;j<M.columns();j++){
				if(j!=0) line.append(",");
				line.append(M.get(i, j));
			}
			line.append("\n");
			fw.write(line.toString());
		}
	}
	
	/**
	 * Save specified dense matrix to specified text file on HDFS.
	 * @param conf the configuratioin object used to get the HDFS instance.
	 * @param pathName the pathname to the output file.
	 * @param M the matrix to be written out.
	 * @throws IOException
	 */
	public static void saveDenseMatrix2D2HDFS(Configuration conf, String pathName, DenseDoubleMatrix2D M) throws IOException{
		List<String> res = new ArrayList<String>();
		for(int i=0;i<M.rows();i++){
			StringBuilder line = new StringBuilder("");
			for(int j=0;j<M.columns();j++){
				if(j!=0) line.append(",");
				line.append(M.get(i, j));
			}
			res.add(line.toString());
		}
		DataOperators.saveTextToHDFS(conf, pathName, res, true);
	}

	/**
	 * Save specified dense matrix to specified text file on HDFS.
	 * @param conf the configuratioin object used to get the HDFS instance.
	 * @param pathName the pathname to the output file.
	 * @param M the matrix to be written out.
	 * @throws IOException
	 */	public static void saveDenseMatrix2D2HDFS(Configuration conf, String pathName, double[][] M) throws IOException{
		List<String> res = new ArrayList<String>();
		for(int i=0;i<M.length;i++){
			StringBuilder line = new StringBuilder("");
			for(int j=0;j<M[0].length;j++){
				if(j!=0) line.append(",");
				line.append(M[i][j]);
			}
			res.add(line.toString());
		}
		DataOperators.saveTextToHDFS(conf, pathName, res, true);
	}
	
	/**
	 * Append specified 1D matrix as text to specified file on HDFS.
	 * @param conf the configuration object used to get the HDFS instance.
	 * @param pathName the pathname to the file to be appended.
	 * @param M the 1D matrix.
	 * @throws IOException
	 */
	public static void appendDenseMatrix1D2HDFS(Configuration conf, String pathName, DenseDoubleMatrix1D M) throws IOException{
		StringBuilder line = new StringBuilder(M.get(0)+"");
		for(int c=1;c<M.size();c++){
			line.append(","+M.get(c));
		}
		DataOperators.appendTextToHDFS(conf, pathName, line.toString());
	}
	
	/**
	 * Save specified 1D matrix to specified text file on HDFS.
	 * @param conf the configuration object use to get the HDFS instance.
	 * @param pathName pathname to the output file.
	 * @param M the 1D matrix.
	 * @param overWrite
	 * @throws IOException
	 */
	public static void saveDenseMatrix1D2HDFS(Configuration conf, String pathName, DenseDoubleMatrix1D M, boolean overWrite) throws IOException{
		List<String> res = new ArrayList<String>();
		for(int i=0;i<M.size();i++){
			res.add(M.get(i)+"");
		}
		DataOperators.saveTextToHDFS(conf, pathName, res, overWrite);
	}
	
	/**
	 * Save specified 1D matrix to specified text file on HDFS.
	 * @param conf the configuration object use to get the HDFS instance.
	 * @param pathName pathname to the output file.
	 * @param M the 1D matrix.
	 * @param overWrite
	 * @throws IOException
	 */	public static void saveDenseMatrix1D2HDFS(Configuration conf, String pathName, double[] M, boolean overWrite) throws IOException{
		List<String> res = new ArrayList<String>();
		for(int i=0;i<M.length;i++){
			res.add(M[i]+"");
		}
		DataOperators.saveTextToHDFS(conf, pathName, res, overWrite);
	}
	
	/**
	 * Convert specified matrix into string form.
	 * <p>
	 * @see DefaultStringifier
	 * @param conf
	 * @param M
	 * @return
	 * @throws IOException
	 */
	public static String matrix2String(Configuration conf, DoubleMatrix2D M) throws IOException{
		conf.set("io.serializations", "org.apache.hadoop.io.serializer.JavaSerialization,org.apache.hadoop.io.serializer.WritableSerialization");
		DefaultStringifier<DoubleMatrix2D> output = new DefaultStringifier<DoubleMatrix2D>(conf, DoubleMatrix2D.class); 
		String str = output.toString(M);
		return str;
	}
	
	/**
	 * Convert specified string into a {@link DoubleMatrix2D}.
	 * <p>
	 * @see DefaultStringifier
	 * @param conf
	 * @param str
	 * @return
	 * @throws IOException
	 */
	public static DoubleMatrix2D String2matrix(Configuration conf, String str) throws IOException{
		conf.set("io.serializations", "org.apache.hadoop.io.serializer.JavaSerialization,org.apache.hadoop.io.serializer.WritableSerialization");
		DefaultStringifier<DoubleMatrix2D> ds = new DefaultStringifier<DoubleMatrix2D>(conf, DoubleMatrix2D.class); 
		DoubleMatrix2D M = (DoubleMatrix2D) ds.fromString(str);
		return M;
	}
}

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

public class MatrixIO {
	
	public static void writeMatrix(String name, LongWritable key, MatrixRowWritable value) throws IOException{
		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.get(conf);
		SequenceFile.Writer wr = SequenceFile.createWriter(fs, conf, new Path(name), LongWritable.class, MatrixRowWritable.class);
		wr.append(key, value);
		wr.sync();
	}
	
	public static void readMatrix(String name) throws IOException{
		FileSystem fs = FileSystem.get(new Configuration());
		SequenceFile.Reader rd = new SequenceFile.Reader(fs, new Path(name), new Configuration());
		LongWritable key = new LongWritable();
		MatrixRowWritable value = new MatrixRowWritable();
		while(rd.next(key, value)){
			double[] vec = value.getVector();
			System.out.print(key.get()+"\t");
			if(value.isSparse()){
				long[] ids = value.getIDs();
				for(int i=0;i<value.getNnz();i++){
					if(i!=0) System.out.println(",");
					System.out.print(ids[i]+","+vec[i]);
				}
			}
			else{
				for(int i=0;i<value.getNnz();i++){
					if(i!=0) System.out.println(",");
					System.out.print(vec[i]);
				}
			}
			System.out.println();
		}
	}
	
	public static DenseDoubleMatrix2D readDenseMatrixFromText(String name) throws IOException{
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
		DenseDoubleMatrix2D M = (DenseDoubleMatrix2D) DoubleFactory2D.dense.make(tmpM.size(), d);
		for(int i=0;i<tmpM.size();i++){
			double[] item = tmpM.get(i);
			for(int j=0;j<d;j++)
				M.setQuick(i, j, item[j]);
		}
		return M;
	}
	
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

	public static void saveDenseMatrix2D2HDFS(Configuration conf, String pathName, double[][] M) throws IOException{
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
	
	public static void appendDenseMatrix1D2HDFS(Configuration conf, String pathName, DenseDoubleMatrix1D M, boolean overWrite) throws IOException{
		StringBuilder line = new StringBuilder(M.get(0)+"");
		for(int c=1;c<M.size();c++){
			line.append(","+M.get(c));
		}
		DataOperators.appendTextToHDFS(conf, pathName, line.toString());
	}
	
	public static void saveDenseMatrix1D2HDFS(Configuration conf, String pathName, DenseDoubleMatrix1D M, boolean overWrite) throws IOException{
		List<String> res = new ArrayList<String>();
		for(int i=0;i<M.size();i++){
			res.add(M.get(i)+"");
		}
		DataOperators.saveTextToHDFS(conf, pathName, res, overWrite);
	}
	
	public static void saveDenseMatrix1D2HDFS(Configuration conf, String pathName, double[] M, boolean overWrite) throws IOException{
		List<String> res = new ArrayList<String>();
		for(int i=0;i<M.length;i++){
			res.add(M[i]+"");
		}
		DataOperators.saveTextToHDFS(conf, pathName, res, overWrite);
	}
	
	public static String matrix2String(Configuration conf, DoubleMatrix2D M) throws IOException{
		conf.set("io.serializations", "org.apache.hadoop.io.serializer.JavaSerialization,org.apache.hadoop.io.serializer.WritableSerialization");
		DefaultStringifier<DoubleMatrix2D> output = new DefaultStringifier<DoubleMatrix2D>(conf, DoubleMatrix2D.class); 
		String str = output.toString(M);
		return str;
	}
	
	public static DoubleMatrix2D String2matrix(Configuration conf, String str) throws IOException{
		conf.set("io.serializations", "org.apache.hadoop.io.serializer.JavaSerialization,org.apache.hadoop.io.serializer.WritableSerialization");
		DefaultStringifier<DoubleMatrix2D> ds = new DefaultStringifier<DoubleMatrix2D>(conf, DoubleMatrix2D.class); 
		DoubleMatrix2D M = (DoubleMatrix2D) ds.fromString(str);
		return M;
	}
}

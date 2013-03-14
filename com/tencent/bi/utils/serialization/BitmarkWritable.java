package com.tencent.bi.utils.serialization;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;

public class BitmarkWritable implements Writable{

	private char K;
	
	private boolean bits[];
	
	@Override
	public void readFields(DataInput in) throws IOException {
		K = in.readChar();
		bits = new boolean[K];
		for(int i=0;i<K;i++){
			bits[i] = in.readBoolean();
		}
	}

	@Override
	public void write(DataOutput output) throws IOException {
		output.writeChar(K);
		for(int i=0;i<K;i++)
			output.writeBoolean(bits[i]);
	}
	
	public void set(boolean bits[]){
		K = (char) bits.length;
		this.bits = new boolean[bits.length];
		for(int i=0;i<K;i++)
			this.bits[i] = bits[i];
	}

}

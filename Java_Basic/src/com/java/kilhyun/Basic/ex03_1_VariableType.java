package com.java.kilhyun.Basic;

public class ex03_1_VariableType {
    /*
		자료형
		- 기본형 : 자바 언어에서 기본적으로 제공해주는 자료형으로 메모리의 크기가 정해져있음
				 ex. 정수형, 문자형, 실수형, 논리형

		- 참조형 : JDK 에서 제공되는 클래스와 프로그래머가 정의하는 클래스이며, 클래스별로 사용하는 크기가 다름
				 ex. String, UDF(User Define Function), ...
	*/
    public static void main(String[] args)
    {
        System.out.println("1. Primary Type");

        // 정수형
        byte vByte = 1;
        short vShort = 2;
        int vInt = 4;
        long vLong = 8;

        // 문자형
        char vChar = 'a';

        // 실수형
        float vFloat = 4;
        double vDouble = 8;

        //논리형
        boolean vBoolean = true;

        System.out.println("Byte Type: " + vByte + ", Size : " + Byte.BYTES);
        System.out.println("Short Type: " + vShort + ", Size : " + Short.BYTES);
        System.out.println("Integer Type: " + vInt + ", Size : " + Integer.BYTES);
        System.out.println("Long Type: " + vLong + ", Size : " + Long.BYTES);
        System.out.println("Char Type: " + vChar + ", Size : " + Character.BYTES);
        System.out.println("Float Type: " + vFloat + ", Size : " + Float.BYTES);
        System.out.println("Double Type: " + vDouble + ", Size : " + Double.BYTES);
        System.out.println("Boolean Type: " + vBoolean + ", Size : " + Boolean.TYPE);

    }

}

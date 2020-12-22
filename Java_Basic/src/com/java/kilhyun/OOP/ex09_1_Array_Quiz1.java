package com.java.kilhyun.OOP;

public class ex09_1_Array_Quiz1 {

    // Q. 문자 배열을 생성하고 출력해보세요
    // 대문자를 A-Z까지 배열에 저장하고 이를 다시 출력하는 프로그램을 생성하시오.
    public static void main(String[] args)
    {
        char[] alphabet = new char[26];
        char letter = 'A';

        for(int i = 0; i < alphabet.length; i++)
        {
//			alphabet[i] = (char)((int)'A' + i);
            alphabet[i] = letter++;
        }

        for(int i = 0; i <alphabet.length; i++)
        {
            System.out.println(alphabet[i]);
        }
    }

}

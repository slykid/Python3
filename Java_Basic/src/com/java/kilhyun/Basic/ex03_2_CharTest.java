package com.java.kilhyun.Basic;

public class ex03_2_CharTest {

    public static void main(String[] args)
    {
        // ASCII Test
        char letter = 'A';  //  A = 65
        System.out.println(letter + ", " + (int)letter);

        letter = (char)((int)letter + 1); // B = 66
        System.out.println(letter + ", " + (int)letter);

        // Unicode Test
        char hangul = '\uAC00';  // 가
        System.out.println(hangul);
    }

}

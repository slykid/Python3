package com.java.kilhyun.Basic;

import java.util.Scanner;

public class ex06_Condition_Quiz2 {

    public static void main(String[] args)
    {
        int month = 0;
        int days = 0;
        Scanner sc = new Scanner(System.in);

        System.out.print("월을 입력하시오(정수만 가능) : ");
        month = sc.nextInt();

        switch(month)
        {
            case 2: days = 28; break;
            case 1: case 3: case 5: case 7: case 8: case 10: case 12: days = 31; break;
            case 4: case 6: case 9: case 11: days = 30; break;
        }

        System.out.println(month + "월의 날짜 수: " + days);
    }

}

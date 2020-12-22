package com.java.kilhyun.Basic;

import java.util.Scanner;

public class ex06_Condition {

    /*
     * if - else 문
     * - 조건식의 결과에 따라 수행문이 실행되는 조건문
     * - 형식
     * if (조건식)
     * {
     * 	수행문;
     * }
     *
     * else if (조건식)
     * {
     * 	수행문;
     * }
     *
     * else
     * {
     * 	수행문
     * }
     *
     * switch - case 문
     * - 결과값에 따라 경우를 나눠 수행문이 실행되는 조건문
     * - 형식
     * switch (조건식)
     * {
     * 		case 조건값:	수행문; break;	// break : 해당부분에서 수행을 중지하겠다는 지시어
     * 		...
     * 		default: 	수행문; 	// 기본값, 삭제 가능
     * }
     */

    public static void main(String[] args)
    {
        String gender = "F";

        // if-else 조건문
        if(gender == "F")
        {
            System.out.println("여성입니다.");
        }
        else
        {
            System.out.println("잘못된 성별입니다.");
        }

        int age = 10;
        int charge = 0;

        if(age < 8)
        {
            charge = 1000;
        }
        else if (age < 14)
        {
            charge = 1500;
        }
        else if (age < 20)
        {
            charge = 2000;
        }
        else
        {
            charge = 3000;
        }

        System.out.println("나이 " + age);
        System.out.println("요금 " + charge);

        //switch - case 조건문
        Scanner sc = new Scanner(System.in);

        System.out.print("성별을 입력하시오: ");
        gender = sc.next();

        switch(gender)
        {
            case "F" : System.out.println("여자입니다."); break;
            case "M" : System.out.println("남자입니다."); break;
        }


    }

}

package com.java.kilhyun.Basic;

import java.util.Scanner;

public class ex07_Loop_Quiz1 {

    // Q. 연산자와 두 수를 변수로 선언한 후 사칙연산이 수행되는 프로그램을 만드시오 (if-else / switch-case 모두 구현)
    public static void main(String[] args)
    {
        int num1 = 0;
        int num2 = 0;
        int res = 0;

        String op = null;

        Scanner sc = new Scanner(System.in);

        System.out.print("숫자 2개를 입력하시오 : ");
        String input = sc.nextLine();
        num1 = Integer.parseInt(input.split(" ")[0]);
        num2 = Integer.parseInt(input.split(" ")[1]);

        System.out.print("연산자를 입력하시오 : ");
        op = sc.next();

        // 1. if-else
        System.out.println("--------------------ver.if-else---------------------------------");

        if(op.equals("+"))
            res = num1 + num2;
        else if(op.equals("-"))
            res = num1 - num2;
        else if (op.equals("*"))
            res = num1 * num2;
        else if (op.equals("/"))
        {
            if(num2 == 0)
                System.out.println("Error : 계산이 불가합니다.");
            else
                res = (int)(num1 / num2);
        }
        else
            System.out.println("Error : 연산자에 오류가 있습니다.\n덧셈, 뺄셈, 곱셈, 나눗셈으로만 입력해야합니다.");

        System.out.println(num1 + " " + op + " " + num2 + " = " + res);

        System.out.println("--------------------ver.switch-case---------------------------------");

        switch(op)
        {
            case "+": res = num1 + num2; break;
            case "-": res = num1 - num2; break;
            case "*": res = num1 * num2; break;
            case "/":
                if(num2 != 0)
                    res = num1 / num2; break;
        }
        System.out.println(num1 + " " + op + " " + num2 + " = " + res);
    }

}

package com.java.kilhyun.Basic;

import java.util.Scanner;

public class ex08_NestedLoop {

    /*
     * 중첩 반복문 (Nested Loop)
     * - 의미 그대로 반복문안에 반복문이 들어있는 코드 구조
     * - "내부 for문 -> 외부 for 문" 순으로 동작함
     */
    public static void main(String[] args)
    {
        // ex. 구구단(2 ~ 9 단)
        int row = 0;
        int col = 0;

        Scanner sc = new Scanner(System.in);

        System.out.print("행과 열의 수를 입력하시오(구분자는 공백): ");
        String input = sc.nextLine();
        row = Integer.parseInt(input.split(" ")[0]);
        col = Integer.parseInt(input.split(" ")[1]);

        // 단 계산
        for(int dan = 2; dan < 9; dan+=row)
        {
            // 배수 계산
            for(int i = 1 ; i < 10; i++)
            {
                // 1행 당 출력할 개수 계산
                for(int j = dan; (j < dan+row) && (j < 10) ; j++)
                {
                    /*
                     *  출력 결과의 행단위로 보았을 때 2 x 1 = 2  3 x 1 = 3 ... 형식이기 때문에
                     *  j 의 초기 값은 단이고, "단 + 1행당 출력 개수" 까지만 출력해야됨
                     */
                    System.out.print(j + " x " + i + " = " + (j*i) + "\t");
                }
                System.out.println();
            }
            System.out.println();

        }

    }

}

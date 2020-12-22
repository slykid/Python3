package com.java.kilhyun.OOP;

public class ex09_5_NDimArray {

    /*
     * 다차원 배열
     * - 2차원 이상의 배열
     * - 공간을 구현할 때 사용됨
     *
     * - 선언
     *   int[][] arr = new int[2][3]  // 행 : 2 , 열 : 3
     */
    public static void main(String[] args)
    {
        int[][] arr = new int[2][3];

        // 선언과 동시에 초기화하기
        int[][] arr2 = { {1,2,3}, {4,5,6}};

        System.out.println(arr2.length);  // 행의 개수를 출력함
        System.out.println(arr2[0].length); // 1번째 행의 개수를 출력함

        System.out.println();

        for(int i = 0; i < arr2.length; i++)
        {
            for(int j = 0; j < arr2[i].length; j++)
                System.out.println(arr2[i][j] + " ");
            System.out.println();
        }

    }

}

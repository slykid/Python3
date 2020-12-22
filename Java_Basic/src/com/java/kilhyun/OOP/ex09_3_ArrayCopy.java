package com.java.kilhyun.OOP;

public class ex09_3_ArrayCopy {

    public static void main(String[] args)
    {
        int[] arr1 = {1, 2, 3, 4, 5};
        int[] arr2 = {10, 20, 30, 40, 50};

        for(int i = 0; i < arr2.length; i++)
        {
            System.out.println(arr2[i]);
        }

        System.out.println();
        System.arraycopy(arr1, 0, arr2, 1, 3);
        // System.arraycopy(src, scope1, dest, scope2, step)
        // src 의 scope1 번째 부터 step 개의 요소를  dest 의 scope2 번째 위치부터 step 개 만큼 복사함

        // 결과 확인
        for(int i = 0; i < arr2.length; i++)
        {
            System.out.println(arr2[i]);
        }
    }

}

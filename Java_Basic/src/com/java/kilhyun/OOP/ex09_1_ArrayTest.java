package com.java.kilhyun.OOP;

public class ex09_1_ArrayTest {

        public static void main(String[] args)
        {

            int[] arr = new int[10];

            int[] arr2 = new int[]{1, 2, 3};
            int[] arr3 = {4, 5, 6};

            for (int i = 0; i < arr.length; i++) {
                System.out.println(arr[i]);
            }

            for (int i = 0, num = 1; i < arr.length; i++, num++) {
                arr[i] = num;
            }

            for (int i = 0; i < arr.length; i++) {
                System.out.println(arr[i]);
            }

            System.out.println("--------------------------------------------------------------------------------------");

            double[] dArr = new double[5];
            dArr[0] = 1.1;
            dArr[1] = 2.2;
            dArr[2] = 3.3;

            double total = 0;
            for (int i = 0; i < dArr.length; i++) {
                total += dArr[i];
            }
            System.out.println(total);

            System.out.println("--------------------------------------------------------------------------------------");

        }

 }



package com.java.kilhyun.Basic;

public class ex07_Loop_Quiz2 {

    // Q. 다이아몬드를 출력하시오.
    public static void main(String[] args)
    {
        int row = 0;
        int star = 0;
        int space = 0;

        for(row = 0; row < 7; row++)
        {
            if(row < 4)
            {
                for(space = 3-row; space > 0 ; space--)
                    System.out.print(" ");
                for(star = space; star < 2*row+1; star++)
                    System.out.print("*");
            }
            else if(row >= 4)
            {
                for(space = 0  ; space < row-3 ; space++)
                    System.out.print(" ");
                for(star = space; star <= 2*(row-space)-space; star++)
                    System.out.print("*");
            }

            System.out.println();

        }

    }


}

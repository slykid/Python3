package com.java.kilhyun.OOP;

public class ex08_2_Singleton_Company {

    private static ex08_2_Singleton_Company instance = new ex08_2_Singleton_Company();

    private ex08_2_Singleton_Company() {}

    //	public Ex_SingletonCompany getInstance() // 일반 메소드임
    public static ex08_2_Singleton_Company getInstance() // 외부에서 가져가도록 하기 위해 앞에 static 울 추가함
    {
        if(instance == null)
        {
            instance = new ex08_2_Singleton_Company();
        }
        return instance;
    }

}

import type { CaseData } from '../types';

export const mockCases: Record<string, CaseData> = {
  "Case_A_vs_B_2023.pdf": {
    type: "judged",
    summary: {
      facts: "The case revolves around a contractual dispute between a software provider (Petitioner) and a client (Respondent). The Petitioner alleges non-payment for delivered services, while the Respondent claims the software did not meet the agreed-upon specifications.",
      petitionerArgs: "The Petitioner presented evidence of the contract, email correspondence confirming delivery, and server logs showing the software's deployment. They argue that the Respondent's claims of malfunction are unsubstantiated and a tactic to avoid payment.",
      respondentArgs: "The Respondent provided internal reports detailing numerous bugs and system failures. They argued that the product was unfit for purpose, constituting a breach of contract, thereby justifying the withholding of payment.",
      reasoning: "The court found that while minor bugs were present, they did not render the software fundamentally unfit for its intended purpose. The contract stipulated a 30-day period for reporting critical failures, which the Respondent failed to adhere to. The court reasoned that the Respondent had accepted the software by using it in a production environment for over three months.",
      decision: "The writ petition is dismissed. The Respondent is ordered to pay the outstanding contractual amount with interest within 60 days."
    }
  },
  "Case_C_vs_D_Pending.pdf": {
    type: "pending",
    summary: {
      facts: "This is a property dispute concerning the ownership of a parcel of land. The Petitioner, Mr. C, claims ownership through ancestral rights. The Respondent, Ms. D, claims ownership based on a registered sale deed from 2010.",
      petitionerArgs: "Mr. C has presented historical land records and witness testimonies suggesting his family has occupied and cultivated the land for several generations.",
      respondentArgs: "Ms. D has presented a legally registered and stamped sale deed, along with tax receipts for the property since 2010.",
      reasoning: null,
      decision: null
    },
    prediction: {
      probabilities: [
        { outcome: "Petition dismissed", probability: "70%" },
        { outcome: "Petition partly allowed (land partitioned)", probability: "25%" },
        { outcome: "Petition allowed", probability: "5%" }
      ],
      precedents: [
        "Civil Appeal No. 1234 (2018): Registered sale deed given precedence over ancestral claims without continuous documented possession.",
        "Land Dispute Case 567 (2021): Ancestral claims upheld where registration documents showed procedural irregularities."
      ]
    }
  }
};